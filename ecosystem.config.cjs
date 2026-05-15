// PM2 ecosystem 配置 — DEV 环境
// 进程名 funasr-server-dev / 端口由 .env 中 FUNASR_SERVER_PORT=8867 决定

module.exports = {
  apps: [
    {
      name: "funasr-server-dev",
      cwd: __dirname,
      script: "run_server.py",
      interpreter: `${__dirname}/venv/bin/python`,

      // DEV 调试时崩了不要无限拉起,失败信号要保留
      autorestart: false,

      // 优雅停机给 funasr 足够时间释放模型/MPS
      kill_timeout: 10000,

      // PM2 日志(只记 stdout/stderr,业务日志走 dev/logs/funasr_server.log)
      out_file: `${__dirname}/logs/pm2-out.log`,
      error_file: `${__dirname}/logs/pm2-error.log`,
      merge_logs: true,
      time: true,

      env: {
        FUNASR_ENV: "dev",
        // 防御 PM2 client/daemon env 中可能残留的临时 TMPDIR(如 MCP 工具的 .ctx-mode-XXX)
        // 一旦原临时目录被清理,fork 出来的进程任何 tempfile 操作都会 ENOENT。
        TMPDIR: "/tmp",
      },
    },
  ],
};
