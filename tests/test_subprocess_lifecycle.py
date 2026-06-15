"""Deterministic tests for the fresh-subprocess command lifecycle.

These tests prove that:
- SubprocessCmdStatus.done does not report completion before communicate()
  has finished collecting both stdout and stderr.
- out_err() returns the complete, stable output tuple once done.
- Fast successful and failing backends preserve their full output.
- Repeated calls do not reuse stale completion state or output.
- Missing response files still surface the collected output.
"""

from __future__ import annotations

import threading
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest

from pyproject_api._frontend import BackendFailed
from pyproject_api._via_fresh_subprocess import SubprocessCmdStatus, SubprocessFrontend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _FakeProcess:
    """A fake Popen whose communicate blocks on an event."""

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode: int | None = None
        self._communicate_started = threading.Event()
        self._release_communicate = threading.Event()
        self._final_returncode = returncode

    def communicate(
        self, input: Any = None, timeout: float | None = None,  # noqa: A002, ARG002
    ) -> tuple[str, str]:
        self._communicate_started.set()
        self.returncode = self._final_returncode
        self._release_communicate.wait(timeout=5)
        return (self._stdout, self._stderr)


class TestSubprocessCmdStatusDoneRace:
    """done must not report True before communicate completes."""

    def test_done_false_while_communicate_running(self) -> None:
        fake = _FakeProcess("out-data", "err-data", returncode=0)
        status = SubprocessCmdStatus(fake)  # type: ignore[arg-type]
        try:
            assert fake._communicate_started.wait(timeout=5), "communicate() never started"
            assert fake.returncode is not None, "test precondition: returncode set"
            assert status.done is False, "done must be False while communicate() is still running"
        finally:
            fake._release_communicate.set()
            status.join(timeout=5)

    def test_done_true_only_after_communicate_completes(self) -> None:
        fake = _FakeProcess("final-out", "final-err", returncode=42)
        status = SubprocessCmdStatus(fake)  # type: ignore[arg-type]
        try:
            assert fake._communicate_started.wait(timeout=5)
            fake._release_communicate.set()
        finally:
            status.join(timeout=5)
        assert status.done is True
        assert status.out_err() == ("final-out", "final-err")


class TestOutErrStability:
    """out_err() must be stable and complete once done is True."""

    def test_out_err_returns_complete_tuple_after_done(self) -> None:
        fake = _FakeProcess("stdout-text", "stderr-text", returncode=0)
        status = SubprocessCmdStatus(fake)  # type: ignore[arg-type]
        fake._release_communicate.set()
        status.join(timeout=5)
        assert status.done is True

        result = status.out_err()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == ("stdout-text", "stderr-text")

    def test_out_err_stable_across_repeated_calls(self) -> None:
        fake = _FakeProcess("stable-out", "stable-err", returncode=0)
        status = SubprocessCmdStatus(fake)  # type: ignore[arg-type]
        fake._release_communicate.set()
        status.join(timeout=5)
        assert status.done is True

        results = [status.out_err() for _ in range(5)]
        assert all(r == ("stable-out", "stable-err") for r in results)

    def test_out_err_never_none_after_done(self) -> None:
        fake = _FakeProcess("x", "y", returncode=1)
        status = SubprocessCmdStatus(fake)  # type: ignore[arg-type]
        fake._release_communicate.set()
        status.join(timeout=5)
        assert status.done is True
        for _ in range(10):
            assert status.out_err() is not None
            assert status.out_err() == ("x", "y")


@pytest.fixture
def local_builder(tmp_path: Path) -> Callable[[str], Path]:
    def _f(content: str) -> Path:
        toml = '[build-system]\nrequires=[]\nbuild-backend = "build_tester"\nbackend-path=["."]'
        (tmp_path / "pyproject.toml").write_text(toml)
        (tmp_path / "build_tester.py").write_text(dedent(content))
        return tmp_path

    return _f


class TestFastBackendOutputPreservation:
    """Fast backend processes must preserve their complete output."""

    def test_fast_successful_backend_preserves_stdout(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            import sys
            def get_requires_for_build_wheel(config_settings=None):
                print("fast-output-line-1")
                print("fast-output-line-2", file=sys.stderr)
                return []
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        result = frontend.get_requires_for_build_wheel()
        assert result.requires == ()
        assert "fast-output-line-1" in result.out
        assert "fast-output-line-2" in result.err

    def test_fast_nonzero_exit_preserves_output_in_backend_failed(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            import sys
            def get_requires_for_build_wheel(config_settings=None):
                print("crash-stdout")
                print("crash-stderr", file=sys.stderr)
                raise RuntimeError("boom")
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        with pytest.raises(BackendFailed) as exc_info:
            frontend.get_requires_for_build_wheel()
        exc = exc_info.value
        assert "crash-stdout" in exc.out
        assert "crash-stderr" in exc.err
        assert exc.exc_type == "RuntimeError"
        assert exc.exc_msg == "boom"

    def test_missing_response_file_preserves_output(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            import sys
            def get_requires_for_build_wheel(config_settings=None):
                print("orphan-stdout")
                print("orphan-stderr", file=sys.stderr)
                sys.exit(42)
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        with pytest.raises(BackendFailed) as exc_info:
            frontend.get_requires_for_build_wheel()
        exc = exc_info.value
        assert "orphan-stdout" in exc.out
        assert "orphan-stderr" in exc.err
        assert exc.code == 42


class TestRepeatedCallsNoStaleState:
    """Repeated fresh-subprocess calls must not reuse stale output or state."""

    def test_sequential_calls_have_independent_output(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            def get_requires_for_build_wheel(config_settings=None):
                print("unique-output-marker")
                return []
            def get_requires_for_build_sdist(config_settings=None):
                print("different-output-marker")
                return []
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

        result1 = frontend.get_requires_for_build_wheel()
        assert "unique-output-marker" in result1.out
        assert "different-output-marker" not in result1.out

        result2 = frontend.get_requires_for_build_sdist()
        assert "different-output-marker" in result2.out
        assert "unique-output-marker" not in result2.out

    def test_sequential_calls_each_produce_fresh_process(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            def get_requires_for_build_wheel(config_settings=None):
                return []
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

        result1 = frontend.get_requires_for_build_wheel()
        result2 = frontend.get_requires_for_build_wheel()
        assert "started backend" in result1.out
        assert "started backend" in result2.out

    def test_failed_then_successful_call_preserves_independence(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        backend_code = """
            import sys
            def get_requires_for_build_wheel(config_settings=None):
                print("fail-output")
                raise ValueError("first call fails")
            def get_requires_for_build_sdist(config_settings=None):
                print("success-output")
                return []
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])

        with pytest.raises(BackendFailed) as exc_info:
            frontend.get_requires_for_build_wheel()
        assert "fail-output" in exc_info.value.out
        assert "success-output" not in exc_info.value.out

        result = frontend.get_requires_for_build_sdist()
        assert "success-output" in result.out
        assert "fail-output" not in result.out


class TestExistingBehaviorCompatibility:
    """Verify that normal calls, missing-command, and BackendFailed still work."""

    def test_normal_call_succeeds(self, local_builder: Callable[[str], Path]) -> None:
        backend_code = """
            def get_requires_for_build_wheel(config_settings=None):
                return ["some-dep"]
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        result = frontend.get_requires_for_build_wheel()
        assert [str(r) for r in result.requires] == ["some-dep"]

    def test_missing_command_raises_backend_failed(
        self,
        local_builder: Callable[[str], Path],
    ) -> None:
        tmp_path = local_builder("")
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        with pytest.raises(BackendFailed) as exc_info:
            frontend.build_wheel(tmp_path)
        assert "has no attribute" in exc_info.value.exc_msg
        assert exc_info.value.exc_type == "MissingCommand"

    def test_backend_failed_carries_output(self, local_builder: Callable[[str], Path]) -> None:
        backend_code = """
            import sys
            def get_requires_for_build_wheel(config_settings=None):
                print("pre-error-output")
                raise RuntimeError("something went wrong")
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        with pytest.raises(BackendFailed) as exc_info:
            frontend.get_requires_for_build_wheel()
        exc = exc_info.value
        assert exc.code == 1
        assert "pre-error-output" in exc.out
        assert exc.exc_type == "RuntimeError"


class TestProcessResourceCleanup:
    """Process resources should be allowed to finish cleanly."""

    def test_process_is_reaped_after_done(self, local_builder: Callable[[str], Path]) -> None:
        backend_code = """
            def get_requires_for_build_wheel(config_settings=None):
                return []
        """
        tmp_path = local_builder(backend_code)
        frontend = SubprocessFrontend(*SubprocessFrontend.create_args_from_folder(tmp_path)[:-1])
        result = frontend.get_requires_for_build_wheel()
        assert result.requires == ()
        assert isinstance(result.out, str)
        assert isinstance(result.err, str)
