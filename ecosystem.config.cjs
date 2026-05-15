// PM2 ecosystem 配置 — PROD 环境
// 进程名 funasr-server / 端口由 .env 中 FUNASR_SERVER_PORT=8767 决定

module.exports = {
  apps: [
    {
      name: "funasr-server",
      cwd: __dirname,
      script: "run_server.py",
      interpreter: `${__dirname}/venv/bin/python`,

      // PROD 自动拉起
      autorestart: true,
      max_memory_restart: "8G",

      // 优雅停机给 funasr 足够时间释放模型/MPS
      kill_timeout: 10000,

      // PM2 日志(只记 stdout/stderr,业务日志走 prod/logs/funasr_server.log)
      out_file: `${__dirname}/logs/pm2-out.log`,
      error_file: `${__dirname}/logs/pm2-error.log`,
      merge_logs: true,
      time: true,

      env: {
        FUNASR_ENV: "prod",
        // 防御 PM2 client/daemon env 中可能残留的临时 TMPDIR(如 MCP 工具的 .ctx-mode-XXX)
        // 一旦原临时目录被清理,fork 出来的进程任何 tempfile 操作都会 ENOENT。
        // 显式钉到系统级稳定路径,确保子进程不会引用易失的临时目录。
        TMPDIR: "/tmp",
      },
    },
  ],
};
