"""
FileBasedProcessPool engine-aware: worker_entry_script 参数 (PR3)

设计目标:
- pool 默认 worker entry 仍是 src/core/worker_process.py (FunASR 路径零侵入)
- 可通过构造参数 worker_entry_script 切到任意 entry (Qwen3 用 src/core/qwen3_worker_process.py)
- _launch_worker_process 把 entry 透传给 subprocess.Popen 的 cmd

覆盖:
1. 默认 worker_entry_script == "src/core/worker_process.py"
2. 自定义 worker_entry_script 在 Popen cmd 中第二个位置 (sys.executable 之后)
3. pool_size 仍可单独指定, 默认走 config.transcription.max_concurrent_tasks
4. 多 worker_id 都用同一个 entry
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.core.file_based_process_pool import FileBasedProcessPool


# ==================== fixtures ====================


@pytest.fixture
def patched_popen(tmp_path, monkeypatch):
    """mock subprocess.Popen + 切到 tmp 任务/日志目录, 避免真启子进程"""
    # 切任务目录到 tmp_path, 避免污染工程
    monkeypatch.chdir(tmp_path)

    fake_proc = MagicMock()
    fake_proc.pid = 99999
    fake_proc.poll.return_value = None  # 没退出

    with patch(
        "src.core.file_based_process_pool.subprocess.Popen", return_value=fake_proc
    ) as mock_popen:
        yield mock_popen, fake_proc


# ==================== 默认行为(FunASR 路径) ====================


class TestDefaultWorkerEntry:
    """worker_entry_script 不传 → 默认 src/core/worker_process.py, FunASR 路径零侵入"""

    def test_default_entry_is_funasr_worker_process(self):
        pool = FileBasedProcessPool(pool_size=2)
        assert pool.worker_entry_script == "src/core/worker_process.py"

    def test_default_pool_size_falls_back_to_config(self):
        """pool_size 不传 → 走 config.transcription.max_concurrent_tasks (与历史一致)"""
        from src.core.config import config

        pool = FileBasedProcessPool()
        assert pool.pool_size == config.transcription.max_concurrent_tasks


# ==================== 自定义 entry 透传 ====================


class TestCustomWorkerEntry:
    """worker_entry_script 自定义 → 传到 Popen cmd"""

    def test_custom_entry_stored_on_instance(self):
        pool = FileBasedProcessPool(
            pool_size=3, worker_entry_script="src/core/qwen3_worker_process.py"
        )
        assert pool.worker_entry_script == "src/core/qwen3_worker_process.py"

    def test_custom_entry_appears_in_popen_cmd(self, patched_popen):
        mock_popen, _ = patched_popen
        pool = FileBasedProcessPool(
            pool_size=3, worker_entry_script="src/core/qwen3_worker_process.py"
        )
        pool._launch_worker_process(worker_id=0)

        # 取第一次 Popen 调用的位置参数 [0] 即 cmd
        assert mock_popen.call_count == 1
        cmd = mock_popen.call_args.args[0]
        # cmd[0] = sys.executable, cmd[1] = worker entry script
        assert cmd[1] == "src/core/qwen3_worker_process.py"
        # 后续参数 --worker-id 0 --task-dir ... 仍然在
        assert "--worker-id" in cmd
        assert "0" in cmd
        assert "--task-dir" in cmd

    def test_default_entry_appears_in_popen_cmd(self, patched_popen):
        """默认 entry 同样落到 cmd[1], 保证 FunASR 路径不变"""
        mock_popen, _ = patched_popen
        pool = FileBasedProcessPool(pool_size=2)
        pool._launch_worker_process(worker_id=1)

        cmd = mock_popen.call_args.args[0]
        assert cmd[1] == "src/core/worker_process.py"

    def test_multiple_workers_share_same_entry(self, patched_popen):
        """同一个 pool 启多个 worker, 都用同一个 entry"""
        mock_popen, _ = patched_popen
        pool = FileBasedProcessPool(
            pool_size=3, worker_entry_script="src/core/qwen3_worker_process.py"
        )
        pool._launch_worker_process(worker_id=0)
        pool._launch_worker_process(worker_id=1)
        pool._launch_worker_process(worker_id=2)

        assert mock_popen.call_count == 3
        for call in mock_popen.call_args_list:
            cmd = call.args[0]
            assert cmd[1] == "src/core/qwen3_worker_process.py"
