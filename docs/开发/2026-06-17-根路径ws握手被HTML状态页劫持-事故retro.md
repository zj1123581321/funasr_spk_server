# 事故 retro：根路径 ws 握手被 HTML 状态页劫持

> 2026-06-17。可观测性仪表盘上线后,客户端报"无法连接到 FunASR 服务器"。
> 已 hotfix(`8022afc`)恢复。本文做根因分析 + 同型测试盲区排查 + 行动项。

## 1. 时间线

```
afbd62b  C3 加 /health + /metrics 端点 (process_request 钩子)
         └ live 测试 test_http_endpoints_live.py 连的是 ws://.../ws (便利路径)
39c8fc7  C7 加 / HTML 状态页 (process_request 对根路径 / 返 HTML)
         └ 引入 bug: 根路径 / 被 HTTP 端点占用
[部署]    prod 重启, /health /metrics / 三端点活体验证通过 (但都是 HTTP 视角)
[事故]    真实客户端连 ws://prod:8767/ (根路径) → 被 HTML 页拦成 HTTP 200
         → ws 握手失败 → "无法连接到服务器"
8022afc  hotfix: process_request 见 Upgrade:websocket 头一律放行 (不管路径)
         └ live 测试补连根路径 / 钉死
```

## 2. 根因(5 whys)

1. **为什么客户端连不上?** 根路径 `/` 被 C7 的 HTML 状态页占用,ws 握手请求被当 HTTP 返回 200。
2. **为什么 process_request 会拦 ws 握手?** 它按 `路径` 路由(`/` → HTML),没先判断"这是不是 ws 升级请求"。ws 握手在 legacy websockets 里也是一个 GET,路径可以是 `/`。
3. **为什么测试没抓到?** C3/C7 的 live 测试连的是 `/ws`(一个"任意非端点路径"),从没连过根路径 `/`——而生产客户端连的正是根路径。
4. **为什么会选 `/ws` 这个便利路径?** 写测试时只想"验证一个非端点路径能走 ws 升级",`/ws` 顺手能跑通,没意识到**生产客户端连的是根路径 `/`**,而根路径恰好和新加的 HTML 端点冲突。
5. **根因**:**测试用了"自造的便利路径"而非"生产真实路径"**,于是测试避开了真正会出问题的代码路径。活体验证也只从 HTTP 视角测了三端点,没从"真实客户端 ws 视角"测根路径。

## 3. 本质模式(可复用的教训)

> **e2e / live 测试必须用生产真实入参(真实路径、真实帧序列、真实数据形态),不能用"任意能跑通的便利值"。**
> 便利值会让测试走一条"演示路径",而生产走另一条"真实路径",bug 藏在两者的差里。

这次具体差异:测试路径 `/ws` ≠ 生产路径 `/`。**正确的不变量**:ws 握手(带 `Upgrade: websocket` 头)无论什么路径都必须穿透 HTTP 端点——已固化进代码 + `test_http_endpoints_live.py` 连根路径钉死 + CLAUDE.md 铁律。

## 4. 同型盲区排查(全套测试)

排查"测试用便利桩 / 生产走不同路径"的同型问题。结论按风险:

### 🔴 高 — chunked 分片上传:逐帧累积逻辑从未被真 ws 帧触达(同型)
- 生产真实路径:大文件/长音频客户端 `upload_mode=chunked` → `_handle_chunked_upload_request` → 多帧 `upload_chunk` 逐帧累积 → 收齐自动 `_finalize_chunked_upload`(`websocket_handler.py:209-211, 551, 690-691`)。
- 测试现状:
  - 所有 e2e 走**单帧 `upload_data`**(整文件一帧,`_ws_e2e_common.py:103-106`),**连名为 `test_single_client_full_chunked_upload` 的测试也是单帧**(`test_qwen3_server_websocket_e2e.py:245`,命名误导)。
  - 唯一碰分片的 unit(`test_websocket_finalize_resilience.py`)**手搓一个已收齐的 session dict**,绕过真实逐帧累积。
  - 全套测试中 `upload_chunk` 只出现在一个协议消息类型字符串列表里,**没有任何测试发过真的逐帧 `upload_chunk` ws 帧**。
- 盲区:`_handle_chunk_upload` 的真实累积(分片计数 / 去重 / 收齐触发 / 乱序重复)零端到端覆盖,而长音频/大批量正是它的主场景。**与本次完全同型**。→ TODOS #24

### 🟡 中 — 单文件 `_handle_upload_data` 错误分支只靠默认 skip 的 e2e
- `_handle_upload_data`(`websocket_handler.py:278-359`)的 size_mismatch / hash_mismatch / missing_file_data / 单文件 queue_full 分支,unit 完全不覆盖,只有默认 skip 的 e2e 走它。日常 CI(unit only)看不到。→ TODOS #24

### 🟢 已堵 / 未发现
- **本次 bug 已堵**:`test_http_endpoints_live.py` 已连根路径 `/` + `/ws` 双验,CLAUDE.md 记铁律。
- **ws 连接路径(整体)**:integration e2e 全部连**根路径**(`_ws_e2e_common.py:85`),即生产真实路径——这块是对的,反而是本次的 unit live 测试连 `/ws` 才是个例(已修)。
- **task_status_batch**:有真入口路由测试 + 真 task 对象,无同型盲区。
- **数据形态**:finalize/batch unit 的 task/session 字段与生产 `FileUploadRequest` 回填一致。

## 5. 行动项

- [x] hotfix:ws 升级一律放行(`8022afc`)
- [x] live 测试连根路径 `/` 钉死
- [x] CLAUDE.md + 设计doc 记"ws 升级一律放行"铁律
- [x] retro + 同型盲区排查(本文)
- [ ] **TODOS #24**:chunked 分片真帧序列 e2e + `_handle_upload_data` 错误分支 unit(高价值,新 session 做)
- [ ] 顺手:`test_single_client_full_chunked_upload` 改名为名副其实(它是单帧),或补一个真 chunked 版

## 6. 一句话

加任何 `process_request` 路由分支前先问:**生产客户端连的真实路径/发的真实帧,我的测试照到了吗?** 这次答案是"没有",代价是一次生产 down。
