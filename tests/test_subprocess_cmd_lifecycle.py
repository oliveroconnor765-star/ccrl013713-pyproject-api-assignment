from __future__ import annotations

from textwrap import dedent
from threading import Event
from typing import TYPE_CHECKING

import pytest

from pyproject_api._frontend import BackendFailed
from pyproject_api._via_fresh_subprocess import SubprocessCmdStatus, SubprocessFrontend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProcess:
    """A controllable stand-in for ``subprocess.Popen`` that mimics CPython
    internal ``communicate()`` behaviour: *returncode* is set **before** the
    method returns its output tuple.

    Two :class:`threading.Event` objects give the test full control over timing:

    * ``returncode_set`` -- signalled once ``returncode`` has been assigned
      inside ``communicate()``, but *before* the output tuple is returned.
    * ``release`` -- ``communicate()`` blocks on this event after setting
      ``returncode``, letting the test observe the intermediate state.
    """

    def __init__(
        self,
        stdout: str,
        stderr: str,
        rc: int,
        returncode_set: Event,
        release: Event,
    ) -> None:
        self.returncode: int | None = None
        self._stdout = stdout
        self._stderr = stderr
        self._rc = rc
        self._returncode_set = returncode_set
        self._release = release

    def communicate(self) -> tuple[str, str]:
        """Mimic CPython: wait() sets returncode, then output is returned."""
        self.returncode = self._rc
        self._returncode_set.set()
        self._release.wait(timeout=5)
        return (self._stdout, self._stderr)


@pytest.fixture
def local_builder(tmp_path: Path) -> Callable[[str], Path]:
    def _f(content: str) -> Path:
        toml = '[build-system]\nrequires=[]\nbuild-backend = "build_tester"\nbackend-path=["."]'
        (tmp_path / "pyproject.toml").write_text(toml)
        (tmp_path / "build_tester.py").write_text(dedent(content))
        return tmp_path

    return _f


# ---------------------------------------------------------------------------
# 1. done must not report True before communicate() output is collected
# ---------------------------------------------------------------------------


def test_done_false_while_communicate_blocked() -> None:
    returncode_set = Event()
    release = Event()
    proc = FakeProcess("hello\n", "world\n", 0, returncode_set, release)

    status = SubprocessCmdStatus(proc)  # type: ignore[arg-type]

    # Wait until communicate() has set returncode but is still blocked.
    assert returncode_set.wait(timeout=5)
    assert proc.returncode is not None, "returncode should be set inside communicate()"

    # The key assertion: done must still be False because output has
    # not been collected yet.
    assert status.done is False

    # Release communicate() and let the thread finish.
    release.set()
    status.join(timeout=5)
    assert not status.is_alive()


def test_done_true_after_communicate_returns() -> None:
    returncode_set = Event()
    release = Event()
    release.set()  # do not block communicate()
    proc = FakeProcess("out", "err", 0, returncode_set, release)

    status = SubprocessCmdStatus(proc)  # type: ignore[arg-type]
    status.join(timeout=5)

    assert status.done is True


# ---------------------------------------------------------------------------
# 2. out_err() is complete, stable, and idempotent after done
# ---------------------------------------------------------------------------


def test_out_err_returns_complete_tuple_after_done() -> None:
    returncode_set = Event()
    release = Event()
    release.set()
    proc = FakeProcess("full stdout", "full stderr", 0, returncode_set, release)

    status = SubprocessCmdStatus(proc)  # type: ignore[arg-type]
    status.join(timeout=5)
    assert status.done is True

    result = status.out_err()
    assert result == ("full stdout", "full stderr")


def test_out_err_is_stable_across_repeated_calls() -> None:
    returncode_set = Event()
    release = Event()
    release.set()
    proc = FakeProcess("repeat", "me", 42, returncode_set, release)

    status = SubprocessCmdStatus(proc)  # type: ignore[arg-type]
    status.join(timeout=5)

    first = status.out_err()
    second = status.out_err()
    third = status.out_err()
    assert first == second == third == ("repeat", "me")


def test_out_err_is_none_before_done() -> None:
    returncode_set = Event()
    release = Event()
    proc = FakeProcess("pending", "pending", 0, returncode_set, release)

    status = SubprocessCmdStatus(proc)  # type: ignore[arg-type]
    assert returncode_set.wait(timeout=5)

    # communicate() has not returned yet, so _out_err is still None.
    assert status.done is False
    assert status.out_err() is None

    release.set()
    status.join(timeout=5)

    # Now it should be available.
    assert status.out_err() == ("pending", "pending")


# ---------------------------------------------------------------------------
# 3. Fast-exit processes preserve complete stdout/stderr
# ---------------------------------------------------------------------------


def test_fast_success_preserves_output(local_builder: Callable[[str], Path]) -> None:
    src = """\
    import sys

    def get_requires_for_build_sdist(config_settings=None):
        print("fast-stdout-marker", flush=True)
        print("fast-stderr-marker", file=sys.stderr, flush=True)
        return []
    """
    tmp_path = local_builder(src)
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
    result = frontend.get_requires_for_build_sdist()

    assert "fast-stdout-marker" in result.out
    assert "fast-stderr-marker" in result.err


def test_fast_nonzero_exit_preserves_output(local_builder: Callable[[str], Path]) -> None:
    src = """\
    import sys

    def get_requires_for_build_sdist(config_settings=None):
        print("error-stdout-marker", flush=True)
        print("error-stderr-marker", file=sys.stderr, flush=True)
        raise SystemExit(7)
    """
    tmp_path = local_builder(src)
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    with pytest.raises(BackendFailed) as ctx:
        frontend.get_requires_for_build_sdist()
    exc = ctx.value
    assert exc.code == 7
    assert "error-stdout-marker" in exc.out
    assert "error-stderr-marker" in exc.err


def test_fast_exit_missing_result_file_preserves_output(
    local_builder: Callable[[str], Path],
) -> None:
    src = """\
    import sys

    def get_requires_for_build_sdist(config_settings=None):
        print("missing-file-stdout", flush=True)
        print("missing-file-stderr", file=sys.stderr, flush=True)
        raise RuntimeError("boom")
    """
    tmp_path = local_builder(src)
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    with pytest.raises(BackendFailed) as ctx:
        frontend.get_requires_for_build_sdist()
    exc = ctx.value
    # Output must be preserved regardless of whether the result file exists.
    assert "missing-file-stdout" in exc.out
    assert "missing-file-stderr" in exc.err


# ---------------------------------------------------------------------------
# 4. Sequential fresh-subprocess calls do not reuse stale state
# ---------------------------------------------------------------------------


def test_sequential_calls_have_independent_output(
    local_builder: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """\
    import os

    def get_requires_for_build_sdist(config_settings=None):
        marker = os.environ.get("OUTPUT_MARKER", "default")
        print(f"stdout-{marker}")
        return []
    """
    tmp_path = local_builder(src)
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    monkeypatch.setenv("OUTPUT_MARKER", "call-1")
    r1 = frontend.get_requires_for_build_sdist()

    monkeypatch.setenv("OUTPUT_MARKER", "call-2")
    r2 = frontend.get_requires_for_build_sdist()

    monkeypatch.setenv("OUTPUT_MARKER", "call-3")
    r3 = frontend.get_requires_for_build_sdist()

    assert "stdout-call-1" in r1.out
    assert "stdout-call-2" in r2.out
    assert "stdout-call-3" in r3.out
    # No cross-contamination.
    assert "call-2" not in r1.out
    assert "call-1" not in r2.out
    assert "call-1" not in r3.out
    assert "call-2" not in r3.out


def test_failure_then_success_no_stale_error(
    local_builder: Callable[[str], Path],
) -> None:
    fail_src = """\
    import sys

    def get_requires_for_build_sdist(config_settings=None):
        print("fail-marker", flush=True)
        raise SystemExit(3)
    """
    tmp_path = local_builder(fail_src)
    frontend_fail = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    with pytest.raises(BackendFailed) as ctx:
        frontend_fail.get_requires_for_build_sdist()
    assert ctx.value.code == 3
    assert "fail-marker" in ctx.value.out

    # Now create a fresh frontend for a successful backend.
    ok_src = """\
    def get_requires_for_build_sdist(config_settings=None):
        print("success-marker")
        return ["some-dep"]
    """
    tmp_path2 = local_builder(ok_src)
    frontend_ok = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path2)[:-1])
    result = frontend_ok.get_requires_for_build_sdist()
    assert "success-marker" in result.out
    assert "fail-marker" not in result.out


def test_multiple_calls_same_frontend_are_independent(
    local_builder: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """\
    import os

    _counter_file = os.environ.get("COUNTER_FILE", "")

    def get_requires_for_build_sdist(config_settings=None):
        if _counter_file and os.path.exists(_counter_file):
            with open(_counter_file) as fh:
                val = int(fh.read())
        else:
            val = 0
        val += 1
        if _counter_file:
            with open(_counter_file, "w") as fh:
                fh.write(str(val))
        print(f"invocation-{val}")
        return []
    """
    tmp_path = local_builder(src)
    counter_file = tmp_path / "counter.txt"

    monkeypatch.setenv("COUNTER_FILE", str(counter_file))
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    r1 = frontend.get_requires_for_build_sdist()
    r2 = frontend.get_requires_for_build_sdist()

    # Each call spawns a fresh process; the counter file proves they are
    # separate invocations and neither reuses the other process state.
    assert "invocation-1" in r1.out
    assert "invocation-2" in r2.out
    assert "invocation-2" not in r1.out
    assert "invocation-1" not in r2.out


# ---------------------------------------------------------------------------
# 5. Backward compatibility: existing patterns still work
# ---------------------------------------------------------------------------


def test_normal_build_sdist(local_builder: Callable[[str], Path]) -> None:
    src = """\
    import tarfile
    from pathlib import Path

    def build_sdist(sdist_directory, config_settings=None):
        name = "compat-1.0.tar.gz"
        path = Path(sdist_directory) / name
        with tarfile.open(str(path), "w:gz") as tar:
            pass
        print("built-sdist")
        return name
    """
    tmp_path = local_builder(src)
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
    result = frontend.build_sdist(tmp_path / "dist")
    assert result.sdist.name == "compat-1.0.tar.gz"
    assert "built-sdist" in result.out


def test_missing_command_raises_backend_failed(
    local_builder: Callable[[str], Path],
) -> None:
    tmp_path = local_builder("")
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    with pytest.raises(BackendFailed) as ctx:
        frontend.build_wheel(tmp_path)
    assert ctx.value.exc_type == "MissingCommand"


def test_missing_backend_module_raises_backend_failed(
    local_builder: Callable[[str], Path],
) -> None:
    tmp_path = local_builder("")
    toml = tmp_path / "pyproject.toml"
    toml.write_text('[build-system]\nrequires=[]\nbuild-backend = "build_tester"')
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

    with pytest.raises(BackendFailed) as ctx:
        frontend.build_wheel(tmp_path / "wheel")
    assert ctx.value.exc_type == "RuntimeError"
    assert "failed to start backend" in ctx.value.err


def test_reuse_backend_is_false_for_subprocess_frontend(
    local_builder: Callable[[str], Path],
) -> None:
    tmp_path = local_builder("")
    frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
    assert frontend._reuse_backend is False  # noqa: SLF001
