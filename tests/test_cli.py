"""CLI wiring (argument parsing and dispatch, no network)."""

from __future__ import annotations

import sys

import pytest

from crossmind import __version__, cli


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in (captured.out + captured.err)


def test_requires_a_command() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy-everything"])
    assert exc.value.code == 2


def test_smoke_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}

    def fake_smoke(_args) -> int:
        called["ok"] = True
        return 0

    monkeypatch.setattr(cli, "_smoke", fake_smoke)
    assert cli.main(["smoke"]) == 0
    assert called.get("ok") is True


def test_serve_dispatch_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve(args) -> int:
        captured.update(host=args.host, port=args.port, reload=args.reload)
        return 0

    monkeypatch.setattr(cli, "_serve", fake_serve)
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9001"]) == 0
    assert captured == {"host": "0.0.0.0", "port": 9001, "reload": False}


def test_module_entrypoint_exists() -> None:
    import importlib

    module = importlib.import_module("crossmind.cli")
    assert hasattr(module, "main")
    assert sys.modules["crossmind.cli"] is module
