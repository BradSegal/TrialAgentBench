"""Real-container qualification for model-generated code execution."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)
from trialagentbench_verifier.executor import DockerCodeSession

from trialagentbench.ports import LLMResponse, ToolCall
from trialagentbench.ports.code_execution import CodeExecutionLimitsV1
from trialagentbench.trialeval.agent import run_agent
from trialagentbench.trialeval.schema import BenchmarkItem
from trialagentbench.util.runtime_context import (
    bounded_provider_context,
    persist_bulky_tool_output,
)

pytestmark = pytest.mark.executor


def _limits() -> CodeExecutionLimitsV1:
    return CodeExecutionLimitsV1(
        timeout_seconds=3,
        memory_mb=512,
        cpu_count=1,
        process_limit=32,
        output_bytes=1024,
        workspace_mb=128,
    )


def _container_removed(name: str) -> bool:
    return subprocess.run(["docker", "inspect", name], capture_output=True, check=False).returncode != 0


def test_executor_isolates_evidence_network_environment_and_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRIALAGENTBENCH_TEST_SECRET", "must-not-leak")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    try:
        assert session.execute_result("x=41").status == "success"
        assert session.execute_result("print(x + 1)").output == "42"
        raw = session.execute_result("import os; os.write(1, b'raw-fd-output')")
        assert raw.output == "raw-fd-output"
        assert session.execute_result("print(open('evidence.txt').read())").output == "original"
        assert (
            session.execute_result(
                "import os; print(os.environ['HOME'], os.environ['XDG_CACHE_HOME'], os.environ['MPLCONFIGDIR'])"
            ).output
            == "/workspace /workspace/.cache /workspace/.config/matplotlib"
        )
        assert (
            session.execute_result(
                "import os; print(*(os.environ[name] for name in "
                "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
                "'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')))"
            ).output
            == "1 1 1 1 1"
        )
        matplotlib_config = session.execute_result("import matplotlib; print(matplotlib.get_configdir())")
        assert matplotlib_config.status == "success"
        assert matplotlib_config.output == "/workspace/.config/matplotlib"
        session.execute_result("open('evidence.txt', 'w').write('changed')")
        assert evidence.read_text(encoding="utf-8") == "original"
        (tmp_path / "new_phase.txt").write_text("phase2", encoding="utf-8")
        assert session.execute_result("print(open('new_phase.txt').read())").output == "phase2"
        assert session.execute_result("import os; print(os.getenv('TRIALAGENTBENCH_TEST_SECRET'))").output == "None"
        host_home = Path("/", "home", "host-only-user").as_posix()
        assert (
            session.execute_result(f"from pathlib import Path; print(Path({host_home!r}).exists())").output == "False"
        )
        network = session.execute_result(
            "import socket\n"
            "try:\n"
            " socket.create_connection(('1.1.1.1', 53), timeout=.2)\n"
            " print('open')\n"
            "except OSError:\n"
            " print('denied')"
        )
        assert network.output == "denied"
        disk = session.execute_result(
            "try:\n"
            " open('fill.bin', 'wb').write(b'x' * (140 * 1024 * 1024))\n"
            " print('unexpected')\n"
            "except OSError:\n"
            " print('bounded')"
        )
        assert disk.output == "bounded"
    finally:
        session.close()
    assert _container_removed(name)


def test_executor_bounds_output_processes_memory_and_time(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    large = session.execute_result("print('x' * 10_000_000)")
    assert large.status == "success"
    assert large.output_truncated is True
    assert len(large.output.encode("utf-8")) < 1100
    pids = session.execute_result(
        "import subprocess\n"
        "children=[]\n"
        "for _ in range(80):\n"
        " try: children.append(subprocess.Popen(['sleep', '2']))\n"
        " except OSError: break\n"
        "print(len(children))\n"
        "for child in children: child.terminate()"
    )
    assert pids.status == "success"
    assert int(pids.output) < 32
    assert session.execute_result("while True: pass").status == "timeout"
    swallowed = session.execute_result(
        "try:\n" " while True: pass\n" "except Exception:\n" " print('timeout swallowed')"
    )
    assert swallowed.status == "timeout"
    assert "timeout swallowed" not in swallowed.output
    recovered = session.execute_result("print('session retained')")
    assert recovered.status == "success"
    assert recovered.output == "session retained"
    session.close()
    assert _container_removed(name)

    memory_session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    memory_name = memory_session._container_name
    memory = memory_session.execute_result("payload = bytearray(b'x' * (900 * 1024 * 1024))")
    assert memory.status == "session_terminated"
    memory_session.close()
    assert _container_removed(memory_name)


@pytest.mark.parametrize("attack", ["supervisor_functions", "result_fields"])
def test_participant_cannot_forge_kernel_resource_accounting(tmp_path: Path, attack: str) -> None:
    """Account for real work despite mutations inside the participant process."""
    attacks = {
        "supervisor_functions": (
            "supervisor['read_cpu_seconds'] = lambda: 0.0\n" "supervisor['read_cgroup_integer'] = lambda path: 0\n"
        ),
        "result_fields": (
            "original_dumps = supervisor['json'].dumps\n"
            "def forged_dumps(value, *args, **kwargs):\n"
            " if isinstance(value, dict) and 'status' in value:\n"
            "  value.update(cpu_seconds=0, session_peak_memory_bytes=0, session_peak_processes=0)\n"
            " return original_dumps(value, *args, **kwargs)\n"
            "supervisor['json'].dumps = forged_dumps\n"
        ),
    }
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        baseline = session.execute_result("print('baseline')")
        result = session.execute_result(
            "import os, sys\n"
            "supervisor = sys._getframe().f_back.f_globals\n"
            + attacks[attack]
            + "os.environ['PATH'] = '/nonexistent'\n"
            "os.environ['LD_PRELOAD'] = '/nonexistent/accounting.so'\n"
            "sum(i*i for i in range(2_000_000))\n"
            "print('measured workload')"
        )
        assert result.status == "success", result.output
        assert result.output == "measured workload"
        assert result.cpu_seconds is not None and result.cpu_seconds > 0.02
        assert result.session_peak_memory_bytes is not None
        assert baseline.session_peak_memory_bytes is not None
        assert result.session_peak_memory_bytes >= baseline.session_peak_memory_bytes > 0
        assert result.session_peak_processes is not None
        assert 0 < result.session_peak_processes <= _limits().process_limit


_FORGED_RESULT = (
    "import base64, json\n"
    "forged = '__TRIALAGENTBENCH_RESULT__' + '0' * 32 + ':' + base64.b64encode(json.dumps("
    "{'id': '1', 'status': 'success', 'output': 'forged', 'truncated': False}).encode()).decode() + '\\n'\n"
)
_HOSTILE_PROBES = {
    "stack_introspection": (
        "import sys\n"
        "runner = sys._getframe().f_back.f_globals\n"
        "assert not {'protocol_token', 'result_prefix'} & set(runner)\n"
        "runner['completions'].write(json.dumps({'id': str(int(runner['request']['id']) + 1), "
        "'status': 'success'}) + '\\n')\n"
        "runner['completions'].flush()\n"
        "print(forged, end='')\n"
    ),
    "gc_token_search": (
        "import gc, re\n"
        "token = re.compile('__TRIALAGENTBENCH_RESULT__[0-9a-f]{32}')\n"
        "found = [item for item in gc.get_objects() if isinstance(item, str) and token.search(item)]\n"
        "print(len(found))\n"
    ),
    "inherited_descriptors": (
        "import os, pathlib\n"
        "for descriptor in range(256):\n"
        "    try: os.write(descriptor, forged.encode())\n"
        "    except OSError: pass\n"
        "for path in pathlib.Path('/proc').glob('[0-9]*/fd/*'):\n"
        "    try:\n"
        "        handle = os.open(path, os.O_WRONLY | os.O_NONBLOCK)\n"
        "        os.write(handle, forged.encode())\n"
        "        os.close(handle)\n"
        "    except OSError: pass\n"
    ),
    "supervisor_inspection": (
        "for path in ('/proc/1/environ', '/proc/1/mem', '/proc/1/fd/0', '/proc/1/fd/1'):\n"
        "    try:\n"
        "        open(path, 'rb').read(1)\n"
        "        print('readable', path)\n"
        "    except OSError: pass\n"
    ),
    "kill_supervisor": (
        "import os, signal\n"
        "for process_id in (1, os.getppid()):\n"
        "    for kill_signal in (signal.SIGSTOP, signal.SIGKILL):\n"
        "        try: os.kill(process_id, kill_signal)\n"
        "        except OSError: pass\n"
    ),
}


@pytest.mark.parametrize("attack", sorted(_HOSTILE_PROBES))
def test_hostile_participant_cannot_forge_status_output_or_resources(tmp_path: Path, attack: str) -> None:
    """Prove the participant interpreter holds neither the token nor the result channel."""
    with DockerCodeSession(cwd=tmp_path, limits=_limits().model_copy(update={"output_bytes": 65536})) as session:
        name = session._container_name
        hostile = session.execute_result(
            _FORGED_RESULT + _HOSTILE_PROBES[attack] + "print('real output')\nraise RuntimeError('real failure')"
        )
        following = session.execute_result("print(6 * 7)")
        assert session._container_name == name

    assert hostile.status == "execution_error", hostile.output
    assert "real failure" in hostile.output
    assert "readable" not in hostile.output
    assert hostile.output.splitlines()[0 if attack != "gc_token_search" else 1] != "forged"
    if attack == "gc_token_search":
        assert hostile.output.splitlines()[0] == "0"
    assert hostile.cpu_seconds is not None and hostile.session_peak_processes is not None
    assert following.status == "success"
    assert following.output == "42"
    assert _container_removed(name)


def test_participant_completion_claim_cannot_hide_a_timeout(tmp_path: Path) -> None:
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        late = session.execute_result(
            "import signal, sys, time\n"
            "signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
            "time.sleep(3.5)\n"
            "print('finished late')"
        )
        recovered = session.execute_result("print('retained')")
    assert late.status == "timeout"
    assert "Execution timed out after 3s." in late.output
    assert recovered.output == "retained"


def test_forged_early_completion_cannot_strand_processes_or_hide_output(tmp_path: Path) -> None:
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        session.execute_result("x = 1")
        forged = session.execute_result(
            "import json, subprocess, sys, time\n"
            "runner = sys._getframe().f_back.f_globals\n"
            "children = [subprocess.Popen(['sleep', '30']) for _ in range(8)]\n"
            "runner['completions'].write(json.dumps({'id': runner['request_id'], 'status': 'success'}) + '\\n')\n"
            "runner['completions'].flush()\n"
            "time.sleep(2)\n"
            "print('must not be reached')"
        )
        processes = session.execute_result("import os; print(len([p for p in os.listdir('/proc') if p.isdigit()]))")
        lost = session.execute_result("print(x)")
    assert forged.status == "execution_error"
    assert "in-memory variables were lost" in forged.output
    assert "must not be reached" not in forged.output
    assert processes.status == "success" and int(processes.output) <= 3
    assert lost.status == "execution_error" and "NameError" in lost.output


def test_unrenderable_participant_traceback_keeps_the_session(tmp_path: Path) -> None:
    """A NameError under a patsy-like formula namespace breaks Python 3.12 traceback suggestions."""
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        session.execute_result("x = 1")
        failed = session.execute_result(
            "class Lookup:\n"
            "    def __getitem__(self, key):\n"
            "        raise KeyError(key)\n"
            "eval('undefined_name', {}, Lookup())"
        )
        retained = session.execute_result("print(x)")
    assert failed.status == "execution_error"
    assert "NameError: name 'undefined_name' is not defined" in failed.output
    assert retained.status == "success" and retained.output == "1"


@pytest.mark.parametrize("record", ["not-base64!", "eyJpZCI6IDF9", "e30="])
def test_malformed_authenticated_result_is_an_execution_error(tmp_path: Path, record: str) -> None:
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        session._lines.put(session._result_prefix + record)
        malformed = session.execute_result("print('discarded')")
        following = session.execute_result("print('clean')")
    assert malformed.status == "execution_error"
    assert malformed.output == "The executor returned a malformed result record."
    assert following.status == "success"
    assert following.output == "clean"


def test_hard_timeout_without_kernel_accounting_remains_a_timeout(tmp_path: Path, monkeypatch) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits().model_copy(update={"timeout_seconds": 1.0}))
    run = subprocess.run
    failures = [1]

    def unavailable(command, **kwargs):
        if "/bin/cat" in command and failures:
            failures.pop()
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="process table exhausted")
        return run(command, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr("trialagentbench_verifier.executor.subprocess.run", unavailable)
            timed_out = session.execute_result(
                "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True: pass"
            )
        recovered = session.execute_result("print('restarted')")
    finally:
        session.close()
    assert timed_out.status == "timeout"
    assert (timed_out.cpu_seconds, timed_out.session_peak_memory_bytes, timed_out.session_peak_processes) == (
        None,
        None,
        None,
    )
    assert recovered.output == "restarted"


def test_session_launches_by_immutable_image_id(tmp_path: Path) -> None:
    with DockerCodeSession(cwd=tmp_path, limits=_limits()) as session:
        launched = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", session._container_name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    assert session.image_id.startswith("sha256:")
    assert launched == session.image_id


def test_failed_kernel_accounting_closes_the_container_without_a_result(tmp_path: Path, monkeypatch) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    container_name = session._container_name
    run = subprocess.run

    def unavailable(command, **kwargs):
        if "/bin/cat" in command:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="accounting unavailable")
        return run(command, **kwargs)

    monkeypatch.setattr("trialagentbench_verifier.executor.subprocess.run", unavailable)
    with pytest.raises(RuntimeError, match="kernel accounting failed"):
        session.execute_result("print('the code finished but usage was not observed')")
    assert _container_removed(container_name)
    assert session.execute_result("print('must not execute')").status == "session_terminated"


def test_executor_retains_scratch_across_hard_timeout_and_close(tmp_path: Path) -> None:
    limits = _limits().model_copy(update={"timeout_seconds": 1.0})
    session = DockerCodeSession(cwd=tmp_path, limits=limits)
    first_name = session._container_name
    assert (
        session.execute_result(
            "from pathlib import Path; Path('scratch/analysis.py').write_text('answer = 42')"
        ).status
        == "success"
    )

    timed_out = session.execute_result(
        "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True: pass"
    )
    assert timed_out.status == "timeout"
    assert "in-memory variables were lost" in timed_out.output
    assert _container_removed(first_name)
    assert session._container_name != first_name
    restored = session.execute_result("print(open('scratch/analysis.py').read())")
    assert restored.status == "success"
    assert restored.output == "answer = 42"

    session.close()
    assert (tmp_path / "scratch" / "analysis.py").read_text(encoding="utf-8") == "answer = 42"


def test_startup_hard_timeout_closes_without_restarting(tmp_path: Path, monkeypatch) -> None:
    """An unresponsive trusted startup must not recursively start another session."""
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name

    def forbidden_restart():
        raise AssertionError("trusted startup retried after its hard timeout")

    monkeypatch.setattr(session, "_restart_after_hard_timeout", forbidden_restart)
    try:
        result = session._execute_result(
            "import signal, time; signal.signal(signal.SIGALRM, signal.SIG_IGN); time.sleep(120)",
            language="python",
            timeout_seconds=0.1,
            restart_on_timeout=False,
        )
        assert result.status == "timeout"
    finally:
        session.close()
    assert _container_removed(name)


def test_executor_enforces_hard_timeout_after_host_sigkill(tmp_path: Path) -> None:
    """The container must bound hostile code even after losing the host's outer watchdog."""
    script = """
import sys
from pathlib import Path
from trialagentbench_verifier.executor import DockerCodeSession
from trialagentbench_verifier.execution_contracts import CodeExecutionLimitsV1
session = DockerCodeSession(cwd=Path(sys.argv[1]), limits=CodeExecutionLimitsV1(
    timeout_seconds=1, memory_mb=512, cpu_count=1, workspace_mb=128, process_limit=32))
print(session._container_name, flush=True)
session.execute_result("import signal, time; signal.signal(signal.SIGALRM, signal.SIG_IGN); "
                       "open('scratch/hanging', 'w').write('active'); time.sleep(120)")
session.close()
"""
    name = None
    with subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True) as host:
        try:
            assert host.stdout is not None
            name = host.stdout.readline().strip()
            assert name.startswith("trialagentbench-")
            deadline = time.monotonic() + 5
            while subprocess.run(
                ["docker", "exec", name, "test", "-f", "/workspace/scratch/hanging"], capture_output=True
            ).returncode:
                assert time.monotonic() < deadline, "participant code did not start"
                time.sleep(0.05)
            host.kill()
            host.wait(timeout=5)
            deadline = time.monotonic() + 8
            while not _container_removed(name):
                assert time.monotonic() < deadline, "orphaned container outlived the participant hard timeout"
                time.sleep(0.1)
        finally:
            if host.poll() is None:
                host.kill()
                host.wait(timeout=5)
            if name is not None and name.startswith("trialagentbench-"):
                subprocess.run(["docker", "stop", "--time", "0", name], capture_output=True, timeout=15)


def test_runtime_context_artifacts_share_the_persistent_session_workspace(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    assert session.execute_result("_path = 'agent-owned'").status == "success"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old interaction"},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "active prompt"},
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=3,
        max_chars=80,
    )
    rendered = persist_bulky_tool_output(
        "captured-" * 100,
        session=session,
        artifact_id="turn-1-1",
        inline_chars=20,
    )
    output_path = "outputs/turn-1-1.txt"

    assert "scratch/" + output_path in rendered
    assert any("scratch/earlier_turns.json" in str(message.get("content")) for message in context)
    assert session.execute_result("print(open('scratch/earlier_turns.json').read())").status == "success"
    retained = session.execute_result(f"print(open('scratch/{output_path}').read())")
    assert retained.status == "success"
    assert retained.output == "captured-" * 100
    assert session.execute_result("print(_path)").output == "agent-owned"

    session.close()
    assert (tmp_path / "scratch" / "earlier_turns.json").is_file()
    assert (tmp_path / "scratch" / output_path).read_text(encoding="utf-8") == "captured-" * 100


def test_executor_rejects_linked_scratch_checkpoint(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    created = session.execute_result("from pathlib import Path; Path('scratch/link').symlink_to('/etc/passwd')")
    assert created.status == "session_terminated"
    session.close()
    assert _container_removed(name)


def test_unicode_normalization_collision_ends_session_not_checkpoint(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    created = session.execute_result(
        "import unicodedata\nfrom pathlib import Path\n"
        "for form in ('NFC', 'NFD'): Path('scratch', unicodedata.normalize(form, 'caf\\u00e9.txt')).write_text(form)"
    )
    # A participant-made collision is the session's outcome, never a checkpoint (infrastructure) error.
    assert created.status == "session_terminated"
    session.close()
    assert _container_removed(name)


def test_executor_rejects_scratch_file_count_amplification_and_recovers(tmp_path: Path) -> None:
    limits = _limits().model_copy(update={"workspace_file_count": 16})
    session = DockerCodeSession(cwd=tmp_path, limits=limits)
    try:
        exhausted = session.execute_result(
            "from pathlib import Path\n"
            "for index in range(32):\n"
            "    Path(f'scratch/empty-{index:03d}').touch()\n"
            "print('must not succeed')"
        )
    finally:
        session.close()

    assert exhausted.status == "session_terminated"
    assert "file-count limit" in exhausted.output
    assert "must not succeed" not in exhausted.output

    clean_session = DockerCodeSession(cwd=tmp_path, limits=limits)
    try:
        clean = clean_session.execute_result("print(6 * 7)")
    finally:
        clean_session.close()
    assert clean.status == "success"
    assert clean.output == "42"


def test_executor_snapshots_live_scratch_without_ending_session(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    try:
        assert (
            session.execute_result(
                "from pathlib import Path; x = 41; Path('scratch/live.txt').write_text('one')"
            ).status
            == "success"
        )
        assert session.snapshot_scratch() == tmp_path / "scratch"
        assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "one"
        assert session.execute_result("print(x + 1)").output == "42"
        assert session.execute_result("Path('scratch/live.txt').write_text('two')").status == "success"
        session.snapshot_scratch()
        assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "two"
    finally:
        session.close()


def test_scratch_checkpoint_stages_outside_the_evidence_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trialagentbench_verifier import executor

    workdir = tmp_path / "item"
    workdir.mkdir()
    validated: list[Path] = []
    original = executor._validate_scratch_tree

    def record(root: Path, limits: CodeExecutionLimitsV1) -> None:
        validated.append(root)
        original(root, limits)

    monkeypatch.setattr(executor, "_validate_scratch_tree", record)
    session = DockerCodeSession(cwd=workdir, limits=_limits())
    try:
        assert session.execute_result("from pathlib import Path; Path('scratch/a.txt').write_text('a')").status == (
            "success"
        )
        session.snapshot_scratch()
    finally:
        session.close()
    staged = [path for path in validated if path.name.endswith(".tmp")]
    # The swap happens beside the mounted workdir, never inside it, and leaves nothing behind.
    assert staged and all(path.parent == tmp_path for path in staged)
    assert [path.name for path in tmp_path.iterdir()] == ["item"]
    assert [path.name for path in workdir.iterdir()] == ["scratch"]
    assert (workdir / "scratch" / "a.txt").read_text(encoding="utf-8") == "a"


def test_scratch_checkpoint_ignores_participant_modules(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    planted = "import sys\nsys.exit(3)\n"
    try:
        # /workspace is the container working directory and PYTHONPATH entry.
        assert (
            session.execute_result(
                "from pathlib import Path\n"
                f"for name in ('tarfile.py', 'sitecustomize.py'): Path('/workspace', name).write_text({planted!r})\n"
                "Path('scratch/live.txt').write_text('kept')"
            ).status
            == "success"
        )
        assert session.snapshot_scratch() == tmp_path / "scratch"
        assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "kept"
    finally:
        session.close()
    assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "kept"


def test_executor_rejects_unsafe_evidence_added_after_start(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("safe", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    try:
        (tmp_path / "unsafe-link.txt").symlink_to(evidence)
        result = session.execute_result("print('must not run')")
        assert result.status == "session_terminated"
        assert "forbidden symlink" in result.output
        assert "must not run" not in result.output
    finally:
        session.close()
    assert _container_removed(name)


def test_executor_rebuilds_evidence_after_same_size_timestamp_tamper(tmp_path: Path) -> None:
    evidence = tmp_path / "data" / "evidence.txt"
    evidence.parent.mkdir()
    evidence.write_text("original", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    try:
        tampered = session.execute_result(
            "import os\n"
            "from pathlib import Path\n"
            "path = Path('data/evidence.txt')\n"
            "original_stat = path.stat()\n"
            "path.write_text('forged!!')\n"
            "os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))\n"
            "Path('data/stale.txt').write_text('stale')\n"
            "print(path.read_text())"
        )
        restored = session.execute_result(
            "from pathlib import Path\n"
            "print(Path('data/evidence.txt').read_text())\n"
            "print(Path('data/stale.txt').exists())"
        )
        clean = session.execute_result("print(6 * 7)")
    finally:
        session.close()

    assert tampered.status == "success"
    assert tampered.output == "forged!!"
    assert restored.status == "success"
    assert restored.output.splitlines() == ["original", "False"]
    assert evidence.read_text(encoding="utf-8") == "original"
    assert clean.status == "success"
    assert clean.output == "42"


def test_executor_kills_double_fork_orphan_before_next_call(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    try:
        spawned = session.execute_result(
            "import os\n"
            "import time\n"
            "from pathlib import Path\n"
            "first_child = os.fork()\n"
            "if first_child == 0:\n"
            "    if os.fork() > 0:\n"
            "        os._exit(0)\n"
            "    os.setsid()\n"
            "    time.sleep(0.4)\n"
            "    Path('scratch/orphan.txt').write_text('escaped')\n"
            "    os._exit(0)\n"
            "os.waitpid(first_child, 0)\n"
            "print('spawned')"
        )
        escaped = session.execute_result(
            "import time\n"
            "from pathlib import Path\n"
            "time.sleep(0.6)\n"
            "print(Path('scratch/orphan.txt').exists())"
        )
        clean = session.execute_result("print(6 * 7)")
    finally:
        session.close()

    assert spawned.status == "success"
    assert spawned.output.endswith("spawned")
    assert escaped.status == "success"
    assert escaped.output == "False"
    assert clean.status == "success"
    assert clean.output == "42"


def test_executor_rejects_link_replacing_projected_evidence(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "evidence.txt").write_text("original", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    try:
        linked = session.execute_result(
            "import shutil\n"
            "from pathlib import Path\n"
            "shutil.rmtree('data')\n"
            "Path('data').symlink_to('scratch', target_is_directory=True)\n"
            "print('linked')"
        )
        rejected = session.execute_result("print('must not execute')")
    finally:
        session.close()

    assert linked.status == "success"
    assert rejected.status == "session_terminated"
    assert "must not execute" not in rejected.output

    clean_session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    try:
        clean = clean_session.execute_result("print(open('data/evidence.txt').read())")
    finally:
        clean_session.close()
    assert clean.status == "success"
    assert clean.output == "original"


def test_executor_mount_excludes_sibling_evaluator_material(tmp_path: Path) -> None:
    assigned = tmp_path / "assigned"
    assigned.mkdir()
    (assigned / "participant.txt").write_text("visible", encoding="utf-8")
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "target.txt").write_text("must-not-mount", encoding="utf-8")

    session = DockerCodeSession(cwd=assigned, limits=_limits())
    try:
        probe = session.execute_result(
            "from pathlib import Path\n"
            "print(Path('participant.txt').read_text())\n"
            "print(Path('../evaluator/target.txt').exists())\n"
            "print(Path('/evaluator/target.txt').exists())"
        )
    finally:
        session.close()

    assert probe.status == "success"
    assert probe.output.splitlines() == ["visible", "False", "False"]


def test_executor_sessions_are_isolated_under_concurrent_load(tmp_path: Path) -> None:
    workdirs = tuple(tmp_path / f"program-{index}" for index in range(4))
    for index, workdir in enumerate(workdirs):
        workdir.mkdir()
        (workdir / "identity.txt").write_text(str(index), encoding="utf-8")

    def execute(index: int) -> tuple[str, str, str]:
        session = DockerCodeSession(cwd=workdirs[index], limits=_limits())
        name = session._container_name
        try:
            result = session.execute_result("print(open('identity.txt').read())")
            return result.status, result.output, name
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=len(workdirs)) as pool:
        results = tuple(pool.map(execute, range(len(workdirs))))

    assert [(status, output) for status, output, _ in results] == [
        ("success", str(index)) for index in range(len(workdirs))
    ]
    assert all(_container_removed(name) for _, _, name in results)


def test_python_and_r_complete_representative_analysis_controls(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "arm": [0, 0, 0, 1, 1, 1],
            "outcome": [0.0, 1.0, float("nan"), 1.0, 1.0, 0.0],
            "time": [2.0, 4.0, 5.0, 1.0, 3.0, 6.0],
            "event": [1, 0, 1, 1, 1, 0],
            "interim_z": [0.5, 1.1, 2.2, 2.2, 2.2, 2.2],
            "efficacy_boundary": [2.0] * 6,
        }
    )
    frame.to_parquet(tmp_path / "analysis.parquet", index=False)
    session = DockerCodeSession(cwd=tmp_path, limits=_limits().model_copy(update={"timeout_seconds": 30.0}))
    try:
        python = session.execute_result(
            "import pandas as pd\n"
            "import statsmodels.formula.api as smf\n"
            "d = pd.read_parquet('analysis.parquet')\n"
            "point = d.loc[d.arm == 1, 'outcome'].mean() - d.loc[d.arm == 0, 'outcome'].mean()\n"
            "model = smf.ols('outcome ~ arm', data=d).fit(cov_type='HC1')\n"
            "se = model.bse['arm']\n"
            "sequential = int((d.interim_z >= d.efficacy_boundary).any())\n"
            "lower = d.outcome.fillna(0).sum() / len(d)\n"
            "upper = (d.outcome.fillna(0).sum() + d.outcome.isna().sum()) / len(d)\n"
            "raw_events = int(d.event.sum())\n"
            "print('|'.join(format(value, '.17g') for value in "
            "(point, se, sequential, lower, upper, raw_events)))"
        )
        r = session.execute_result(
            "library(trialagentbenchR)\n"
            "library(sandwich)\n"
            "library(survival)\n"
            "library(survey)\n"
            "d <- read_trialagentbench_parquet('analysis.parquet')\n"
            "point <- mean(d$outcome[d$arm == 1], na.rm=TRUE) - mean(d$outcome[d$arm == 0], na.rm=TRUE)\n"
            "model <- lm(outcome ~ arm, data=d)\n"
            "se <- sqrt(diag(vcovHC(model, type='HC1')))[['arm']]\n"
            "sequential <- as.integer(any(d$interim_z >= d$efficacy_boundary))\n"
            "lower <- sum(d$outcome, na.rm=TRUE) / nrow(d)\n"
            "upper <- (sum(d$outcome, na.rm=TRUE) + sum(is.na(d$outcome))) / nrow(d)\n"
            "raw_events <- sum(d$event)\n"
            "invisible(coxph(Surv(time, event) ~ arm, data=d))\n"
            "invisible(svydesign(ids=~1, data=d, weights=~rep(1, nrow(d))))\n"
            "cat(paste(format(c(point, se, sequential, lower, upper, raw_events), "
            "digits=17, scientific=FALSE, trim=TRUE), collapse='|'))",
            language="r",
        )
    finally:
        session.close()

    assert python.status == "success"
    assert r.status == "success"
    python_values = [float(value) for value in python.output.split("|")]
    r_values = [float(value) for value in r.output.split("|")]
    assert python_values == pytest.approx(r_values, rel=1e-12, abs=1e-12)
    assert python_values[2:] == pytest.approx([1.0, 0.5, 2.0 / 3.0, 4.0])


def test_r_execution_is_bounded_and_stateful_after_timeout(tmp_path: Path) -> None:
    # Each R request starts Rscript and restores its workspace; use the same
    # bounded allowance as the other real-container probes, including startup.
    limits = _limits()
    session = DockerCodeSession(cwd=tmp_path, limits=limits)
    try:
        initialized = session.execute_result("x <- 42", language="r")
        assert initialized.status == "success", initialized.output
        timed_out = session.execute_result("while (TRUE) {}", language="r")
        recovered = session.execute_result("cat(x)", language="r")
    finally:
        session.close()

    assert timed_out.status == "timeout"
    assert timed_out.language == "r"
    assert recovered.status == "success"
    assert recovered.output == "42"


def test_r_hostile_probe_cannot_mutate_input_leak_secret_or_leave_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRIALAGENTBENCH_TEST_SECRET", "must-not-leak")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    try:
        probe = session.execute_result(
            "cat(Sys.getenv('TRIALAGENTBENCH_TEST_SECRET', unset='absent'), '\\n')\n"
            "cat(file.create('evidence.txt'), '\\n')\n"
            "system(\"sh -c 'sleep 30 >/dev/null 2>&1 &'\")\n"
            "connection <- socketConnection(host='1.1.1.1', port=53, open='r+', timeout=0.2, blocking=TRUE)\n",
            language="r",
        )
        clean = session.execute_result("cat('clean')", language="r")
        processes = subprocess.run(
            ["docker", "top", name, "-eo", "pid,comm"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        session.close()

    assert probe.status == "execution_error"
    assert "absent" in probe.output
    assert evidence.read_text(encoding="utf-8") == "original"
    assert "sleep" not in processes
    assert clean.status == "success"
    assert clean.output == "clean"
    assert _container_removed(name)


def test_trialeval_agent_loop_uses_the_isolated_executor(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    data = visible / "data"
    hidden = tmp_path / "hidden"
    data.mkdir(parents=True)
    hidden.mkdir()
    (data / "analysis.parquet").write_bytes(b"fixture")
    write_minimal_trialeval_release_dictionaries(tmp_path)
    item = BenchmarkItem(
        item_id="executor-smoke",
        trial_name="executor-smoke",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C4",
        data_preparation="analysis_ready",
        visible_dir=visible,
        data_dir=data,
        task={"target_ids": ["effect"]},
        submission_contract=minimal_participant_output_contract("executor-smoke"),
        suite_dir=tmp_path,
    )

    class StubProvider:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, **_: object) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="write-workspace",
                            name="write_workspace_file",
                            arguments=json.dumps(
                                {
                                    "path": "analysis.py",
                                    "content": "value = 6 * 7\nprint(value)\n"
                                    "from interface.reporting import targets, report\n"
                                    "report([dict(target_id=targets()[0], estimate=0, lower=-1, upper=1)])\n",
                                }
                            ),
                        ),
                        ToolCall(
                            id="read-workspace",
                            name="read_workspace_file",
                            arguments='{"path":"scratch/analysis.py","start_line":1,"end_line":2}',
                        ),
                        ToolCall(
                            id="invalid-execute",
                            name="execute_code",
                            arguments="{not-json",
                        ),
                        ToolCall(
                            id="execute",
                            name="execute_code",
                            arguments='{"code":"value = 6 * 7\\nprint(value)"}',
                        ),
                    ]
                )
            submission = (
                "# Clinical study analysis\n\n"
                "The randomized comparison estimates the prespecified treatment effect. "
                "The executed estimate was 0.0 days (95% confidence interval -1.0 to 1.0) "
                "using the supplied analysis data. This fixture verifies the isolated executor; "
                "the synthetic byte fixture does not support substantive interpretation."
            )
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="submit",
                        name="submit_response",
                        arguments=json.dumps({"content": submission}),
                    )
                ]
            )

    provider = StubProvider()
    conversation_path = tmp_path / "conversation.json"
    event_path = tmp_path / "events.jsonl"
    result = run_agent(
        item,
        provider,
        max_turns=4,
        verbose=False,
        conversation_log_path=conversation_path,
        event_log_path=event_path,
    )

    assert result["status"] == "success"
    assert provider.calls == 2
    assert any(row.get("tool") == "write_workspace_file" for row in result["conversation"])
    assert any(
        row.get("tool") == "read_workspace_file" and "0002: print(value)" in str(row.get("output"))
        for row in result["conversation"]
    )
    assert any(
        row.get("tool") == "execute_code" and str(row.get("output", "")).splitlines()[0] == "42"
        for row in result["conversation"]
        if str(row.get("output", "")).splitlines()
    )
    assert any(
        row.get("tool_call_id") == "invalid-execute" and str(row.get("output", "")).startswith("Tool input rejected:")
        for row in result["conversation"]
    )
    assert [event["event_index"] for event in result["events"]] == list(range(len(result["events"])))
    assert any(event["status"] == "invalid" for event in result["events"])
    assert any(event["event_type"] == "code_execution" for event in result["events"])
    assert any(
        event["event_type"] == "file_inspection" and event["file_accessed"] == "scratch/analysis.py"
        for event in result["events"]
    )
    submission_events = [event for event in result["events"] if event["event_type"] == "submission"]
    assert len(submission_events) == 1
    assert submission_events[0]["status"] == "observed"
    assert json.loads(conversation_path.read_text(encoding="utf-8")) == result["conversation"]
    persisted_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert persisted_events == result["events"]
