#!/usr/bin/env bash
# Qwen3-Diarize 引擎模型权重下载脚本
#
# 模型清单:
#   ASR 引擎 (Qwen3-ASR-1.7B GGUF, 总 ~2.1GB)
#     - qwen3_asr_encoder_frontend.onnx      ~24 MB
#     - qwen3_asr_encoder_backend.onnx       ~611 MB
#     - qwen3_asr_llm.gguf                   ~1.47 GB
#   Diarize 引擎 (sherpa-onnx, 总 ~45MB)
#     - pyannote-segmentation-3.0/model.onnx       ~6 MB     (fp32 主用)
#     - pyannote-segmentation-3.0/model.int8.onnx  ~1.5 MB   (int8 备选)
#     - nemo-titanet-small/embedding.onnx          ~38 MB
#   词级时间戳 (MMS-300M CTC-FA, deskpai ONNX)
#     - ctc_forced_aligner/model.onnx              ~1.2 GB   (word_align, 可选)
#
# 模式:
#   1. --from-prod  : 从 ~/Production/qwen_asr_server/models 本地拷贝(本机 dev 首选)
#   2. --from-url   : 从 GitHub / HuggingFace 下载
#   3. (无参数)     : 优先 --from-prod, 失败回退 --from-url
#
# 下载源:
#   - Qwen3-ASR GGUF: HaujetZhao/CapsWriter-Offline GitHub release 'models' tag
#     (URL 由 QWEN3_ASR_URL_BASE 控制, 默认 GitHub)
#   - sherpa diarize 模型: csukuangfj HuggingFace 镜像
#     (URL 由 SHERPA_PYANNOTE_URL / SHERPA_NEMO_URL 控制)
#
# 用法:
#   bash scripts/download_qwen3_models.sh [--from-prod | --from-url]
#
# 幂等性: 文件已存在且大小匹配则跳过

set -euo pipefail

# ==================== 配置 ====================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${QWEN3_MODELS_DIR:-${PROJECT_ROOT}/models/qwen3_diarize}"
PROD_MIRROR="${QWEN3_PROD_MIRROR:-${HOME}/Production/qwen_asr_server/models}"
# spike 目录里有 sherpa diarize 模型(prod 不用 diarize, 所以这里独立指向 spike)
SPIKE_SHERPA_MIRROR="${QWEN3_SPIKE_SHERPA_MIRROR:-${PROJECT_ROOT}/spikes/qwen3_diarize/models/sherpa}"

# Qwen3-ASR 子目录
ASR_DIR="${MODELS_ROOT}/Qwen3-ASR-1.7B"
SHERPA_DIR="${MODELS_ROOT}/sherpa"
PYANNOTE_DIR="${SHERPA_DIR}/pyannote-segmentation-3.0"
NEMO_DIR="${SHERPA_DIR}/nemo-titanet-small"

# 下载源 (可被环境变量覆盖)
QWEN3_ASR_URL_BASE="${QWEN3_ASR_URL_BASE:-https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models}"
SHERPA_PYANNOTE_URL_BASE="${SHERPA_PYANNOTE_URL_BASE:-https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main}"
SHERPA_NEMO_URL_BASE="${SHERPA_NEMO_URL_BASE:-https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models}"

# 文件清单 (name | dest_subpath | expected_size_bytes | source_path_in_prod)
# size 用近似值, 容差 1MB; -1 表示不校验
FILES=(
    "qwen3_asr_encoder_frontend.onnx|Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx|24063978|Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx"
    "qwen3_asr_encoder_backend.onnx|Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx|611276553|Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx"
    "qwen3_asr_llm.gguf|Qwen3-ASR-1.7B/qwen3_asr_llm.gguf|1471800896|Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_llm.gguf"
)

# ==================== 日志 ====================
log()  { echo "[$(date +%H:%M:%S)] $*"; }
err()  { echo "[$(date +%H:%M:%S)] ❌ $*" >&2; }
ok()   { echo "[$(date +%H:%M:%S)] ✓ $*"; }
skip() { echo "[$(date +%H:%M:%S)] ⊘ $*"; }

# ==================== 工具 ====================
file_size() {
    if [[ ! -f "$1" ]]; then echo 0; return; fi
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0
}

# 跨平台 size 校验: 容差 1MB, expected=-1 时不校验
size_ok() {
    local actual_size="$1" expected="$2"
    [[ "$expected" == "-1" ]] && return 0
    local diff=$(( actual_size > expected ? actual_size - expected : expected - actual_size ))
    (( diff <= 1048576 ))
}

# ==================== 模式: prod 镜像复制 ====================
copy_from_prod() {
    log "尝试从本地 prod 镜像复制: ${PROD_MIRROR}"
    if [[ ! -d "${PROD_MIRROR}" ]]; then
        err "prod 镜像不存在: ${PROD_MIRROR}"
        return 1
    fi

    local fail=0
    for entry in "${FILES[@]}"; do
        IFS='|' read -r name dest_sub expected_size prod_sub <<< "$entry"
        local src="${PROD_MIRROR}/${prod_sub}"
        local dst="${MODELS_ROOT}/${dest_sub}"

        if [[ -f "$dst" ]] && size_ok "$(file_size "$dst")" "$expected_size"; then
            skip "${name} 已存在且大小匹配"
            continue
        fi
        if [[ ! -f "$src" ]]; then
            err "prod 镜像缺文件: ${src}"
            fail=1
            continue
        fi
        mkdir -p "$(dirname "$dst")"
        log "复制 ${name} (${expected_size} bytes)"
        cp "$src" "$dst"
        if size_ok "$(file_size "$dst")" "$expected_size"; then
            ok "${name} 复制完成"
        else
            err "${name} 大小校验失败 actual=$(file_size "$dst") expected=${expected_size}"
            fail=1
        fi
    done

    # sherpa 模型: prod 不用 diarize, 优先从 spike 镜像拷贝
    if [[ -d "${SPIKE_SHERPA_MIRROR}" ]]; then
        log "sherpa diarize 模型从 spike 镜像复制: ${SPIKE_SHERPA_MIRROR}"
        declare -a SPIKE_SHERPA=(
            "pyannote-segmentation-3.0/model.onnx|${PYANNOTE_DIR}/model.onnx|5992913"
            "pyannote-segmentation-3.0/model.int8.onnx|${PYANNOTE_DIR}/model.int8.onnx|1540506"
            "nemo-titanet-small/embedding.onnx|${NEMO_DIR}/embedding.onnx|-1"
        )
        for entry in "${SPIKE_SHERPA[@]}"; do
            IFS='|' read -r sub dst expected_size <<< "$entry"
            local src="${SPIKE_SHERPA_MIRROR}/${sub}"
            if [[ -f "$dst" ]] && size_ok "$(file_size "$dst")" "$expected_size"; then
                skip "sherpa/${sub} 已存在且大小匹配"
                continue
            fi
            if [[ ! -f "$src" ]]; then
                err "spike sherpa 镜像缺: ${src}"
                fail=1
                continue
            fi
            mkdir -p "$(dirname "$dst")"
            log "复制 sherpa/${sub}"
            cp "$src" "$dst"
            ok "sherpa/${sub} 复制完成"
        done
    else
        err "spike sherpa 镜像不存在: ${SPIKE_SHERPA_MIRROR} — diarize 模型未补齐, 用 '--from-url' 兜底"
        fail=1
    fi
    return $fail
}

# ==================== 模式: URL 下载 ====================
download_from_url() {
    log "从 URL 下载模型"
    if ! command -v curl &>/dev/null; then
        err "需要 curl, 请先安装"
        return 1
    fi

    local fail=0
    for entry in "${FILES[@]}"; do
        IFS='|' read -r name dest_sub expected_size _prod_sub <<< "$entry"
        local dst="${MODELS_ROOT}/${dest_sub}"

        if [[ -f "$dst" ]] && size_ok "$(file_size "$dst")" "$expected_size"; then
            skip "${name} 已存在且大小匹配"
            continue
        fi

        # Qwen3-ASR 模型: GitHub release 上的 asset 名通常是单文件直传
        # CapsWriter-Offline 实际可能用 zip 打包, 这里假设直传(若失效需调整 URL_BASE)
        local url="${QWEN3_ASR_URL_BASE}/${name}"
        mkdir -p "$(dirname "$dst")"
        log "下载 ${name} from ${url}"
        if ! curl -L --fail --retry 3 --retry-delay 5 -o "${dst}.partial" "$url"; then
            err "下载失败: ${url}"
            err "  请手动从 https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models 获取"
            err "  并放置到 ${dst}"
            rm -f "${dst}.partial"
            fail=1
            continue
        fi
        mv "${dst}.partial" "$dst"
        if size_ok "$(file_size "$dst")" "$expected_size"; then
            ok "${name} 下载完成"
        else
            err "${name} 大小校验失败 actual=$(file_size "$dst") expected=${expected_size}"
            fail=1
        fi
    done

    # ==================== Sherpa diarize 模型 ====================
    # pyannote-segmentation-3.0 (单文件直传)
    declare -a SHERPA_FILES=(
        "${PYANNOTE_DIR}/model.onnx|${SHERPA_PYANNOTE_URL_BASE}/model.onnx|5992913"
        "${PYANNOTE_DIR}/model.int8.onnx|${SHERPA_PYANNOTE_URL_BASE}/model.int8.onnx|1540506"
    )
    for entry in "${SHERPA_FILES[@]}"; do
        IFS='|' read -r dst url expected_size <<< "$entry"
        local name="$(basename "$(dirname "$dst")")/$(basename "$dst")"
        if [[ -f "$dst" ]] && size_ok "$(file_size "$dst")" "$expected_size"; then
            skip "${name} 已存在且大小匹配"
            continue
        fi
        mkdir -p "$(dirname "$dst")"
        log "下载 ${name} from ${url}"
        if ! curl -L --fail --retry 3 --retry-delay 5 -o "${dst}.partial" "$url"; then
            err "下载失败: ${url}"
            rm -f "${dst}.partial"
            fail=1
            continue
        fi
        mv "${dst}.partial" "$dst"
        ok "${name} 下载完成"
    done

    # NeMo TitaNet small (sherpa-onnx 用 tar.bz2 打包, 简化: 用环境变量给整 tarball URL)
    # 默认走 sherpa-onnx GitHub release, asset 名一般为
    #   sherpa-onnx-nemo-speaker-models-titanet-small-en-NN.tar.bz2
    # 这里不解释具体 release tag, 留环境变量 SHERPA_NEMO_TARBALL_URL 由运维指定
    local nemo_dst="${NEMO_DIR}/embedding.onnx"
    if [[ -f "$nemo_dst" ]] && size_ok "$(file_size "$nemo_dst")" 38000000; then
        skip "nemo-titanet-small/embedding.onnx 已存在"
    else
        if [[ -n "${SHERPA_NEMO_TARBALL_URL:-}" ]]; then
            log "下载 sherpa NeMo TitaNet small from ${SHERPA_NEMO_TARBALL_URL}"
            mkdir -p "${NEMO_DIR}"
            local tarball="${NEMO_DIR}/_titanet.tar.bz2"
            if curl -L --fail --retry 3 -o "$tarball" "${SHERPA_NEMO_TARBALL_URL}"; then
                tar xjf "$tarball" -C "${NEMO_DIR}" --strip-components=1
                rm -f "$tarball"
                ok "nemo-titanet-small 解压完成"
            else
                err "下载失败: ${SHERPA_NEMO_TARBALL_URL}"
                fail=1
            fi
        else
            err "缺 NeMo TitaNet small embedding.onnx, 请设置 SHERPA_NEMO_TARBALL_URL"
            err "  或参考 https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html"
            err "  下载并解压到 ${NEMO_DIR}/"
            fail=1
        fi
    fi

    return $fail
}

# ==================== 校验 ====================
verify_all() {
    log "校验所有模型文件"
    local fail=0
    for entry in "${FILES[@]}"; do
        IFS='|' read -r name dest_sub expected_size _prod_sub <<< "$entry"
        local dst="${MODELS_ROOT}/${dest_sub}"
        if [[ -f "$dst" ]] && size_ok "$(file_size "$dst")" "$expected_size"; then
            ok "${name} ($(file_size "$dst") bytes)"
        else
            err "${name} 缺失或大小异常: ${dst}"
            fail=1
        fi
    done
    local sherpa_files=(
        "${PYANNOTE_DIR}/model.onnx:5992913"
        "${PYANNOTE_DIR}/model.int8.onnx:1540506"
        "${NEMO_DIR}/embedding.onnx:-1"
    )
    for entry in "${sherpa_files[@]}"; do
        IFS=':' read -r dst expected_size <<< "$entry"
        if [[ -f "$dst" ]]; then
            ok "$(basename "$(dirname "$dst")")/$(basename "$dst") ($(file_size "$dst") bytes)"
        else
            err "缺: $dst"
            fail=1
        fi
    done
    return $fail
}

# ==================== word_align MMS CTC-FA 模型 ====================
# 词级时间戳用 MMS-300M CTC-FA (deskpai ctc_forced_aligner ONNX, ~1.2GB).
# 优先从 deskpai 运行时下载位置 ~/ctc_forced_aligner/model.onnx 拷贝 (PoC 已下),
# 缺失则从 HuggingFace 直下 MODEL_URL. 落到 config 默认路径
# ${MODELS_ROOT}/ctc_forced_aligner/model.onnx (不走运行时下载, 避免首请求卡 1.2GB).
WORD_ALIGN_DST="${MODELS_ROOT}/ctc_forced_aligner/model.onnx"
WORD_ALIGN_HOME_CACHE="${HOME}/ctc_forced_aligner/model.onnx"
WORD_ALIGN_URL="${WORD_ALIGN_URL:-https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx}"

fetch_word_align_model() {
    log "准备 word_align MMS CTC-FA 模型: ${WORD_ALIGN_DST}"
    if [[ -f "${WORD_ALIGN_DST}" ]] && (( $(file_size "${WORD_ALIGN_DST}") > 100000000 )); then
        skip "word_align MMS 模型已存在"
        return 0
    fi
    mkdir -p "$(dirname "${WORD_ALIGN_DST}")"
    if [[ -f "${WORD_ALIGN_HOME_CACHE}" ]]; then
        log "从 deskpai 缓存复制: ${WORD_ALIGN_HOME_CACHE}"
        cp "${WORD_ALIGN_HOME_CACHE}" "${WORD_ALIGN_DST}"
        ok "word_align MMS 模型复制完成 ($(file_size "${WORD_ALIGN_DST}") bytes)"
        return 0
    fi
    log "deskpai 缓存缺失, 从 HuggingFace 下载: ${WORD_ALIGN_URL}"
    if curl -L --fail --retry 3 --retry-delay 5 -o "${WORD_ALIGN_DST}.partial" "${WORD_ALIGN_URL}"; then
        mv "${WORD_ALIGN_DST}.partial" "${WORD_ALIGN_DST}"
        ok "word_align MMS 模型下载完成"
        return 0
    fi
    rm -f "${WORD_ALIGN_DST}.partial"
    err "word_align MMS 模型获取失败 (词级时间戳功能不可用, 不影响 ASR/diarize)"
    return 1
}

# ==================== 主流程 ====================
mkdir -p "${MODELS_ROOT}"

MODE="${1:-auto}"

case "$MODE" in
    --from-prod)
        copy_from_prod
        ;;
    --from-url)
        download_from_url
        ;;
    --verify)
        verify_all
        exit $?
        ;;
    --word-align)
        # 仅补 word_align MMS 模型 (词级时间戳)
        fetch_word_align_model
        exit $?
        ;;
    auto|"")
        # 优先 prod 镜像, 缺失则提示用 --from-url
        if [[ -d "${PROD_MIRROR}" ]]; then
            log "检测到 prod 镜像, 走 --from-prod 模式"
            copy_from_prod || {
                err "prod 镜像复制部分失败, 用 '--from-url' 从网络下载补齐"
                exit 1
            }
        else
            log "无 prod 镜像, 走 --from-url 模式"
            download_from_url || exit 1
        fi
        ;;
    *)
        err "未知参数: $MODE"
        echo "用法: $0 [--from-prod | --from-url | --verify]" >&2
        exit 2
        ;;
esac

# word_align MMS 模型 (词级时间戳): 失败不阻塞主流程 (ASR/diarize 仍可用)
fetch_word_align_model || true

echo
log "最终校验"
verify_all
