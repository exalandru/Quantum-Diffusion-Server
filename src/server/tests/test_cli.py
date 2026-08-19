"""The `qds` command, and the environment-variable contract underneath it.

Two properties are worth holding onto here. The dispatcher must hand a
subcommand its arguments *untouched* — it knows the names of the subcommands and
nothing about what they accept, which is the only reason there is no second copy
of every option list to drift. And a `QDS_SERVER_*` variable must win over the
`MFLUX_SERVER_*` one it replaced, while the old spelling keeps being read: a
stale `MFLUX_SERVER_CONFIG` in a launch agent that is silently ignored would
start the server on packaged defaults with nothing in the log to say why.
"""

from __future__ import annotations

import json

import pytest

from qds import cli, env

# ── The dispatcher ─────────────────────────────────────────────────────────


def test_no_arguments_prints_the_commands(capsys):
    assert cli.main([]) == 0
    printed = capsys.readouterr().out
    for command in ("serve", "fetch", "prequantize", "import", "status"):
        assert command in printed


def test_an_unknown_command_is_refused_rather_than_guessed(capsys):
    assert cli.main(["bogus"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_the_version_is_the_packages(capsys):
    from qds import __version__

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("serve", "qds.app.main"),
        ("fetch", "qds.fetch.main"),
        ("prequantize", "qds.prequantize.main"),
        ("import", "qds.import_cli.main"),
    ],
)
def test_each_command_reaches_its_own_main_with_its_own_arguments(monkeypatch, command, target):
    """The dispatcher forwards the tail verbatim, including things it cannot parse."""
    seen: list[list[str]] = []
    monkeypatch.setattr(target, lambda argv=None: (seen.append(list(argv or [])), 0)[1])

    tail = ["--some-flag", "value", "--", "-x"]
    assert cli.main([command, *tail]) == 0
    assert seen == [tail]


def test_the_exit_status_of_a_command_is_the_exit_status_of_qds(monkeypatch):
    monkeypatch.setattr("qds.fetch.main", lambda argv=None: 3)
    assert cli.main(["fetch", "anything"]) == 3


def test_fetch_with_neither_a_model_nor_status_says_so(capsys):
    """`qds fetch` alone is a usage error, not a silent no-op.

    Regression: splitting the parser out of `main` left `parser.error(...)` with
    nothing named `parser` in scope, so this path raised `NameError` instead of
    printing usage — and no test went down it.
    """
    from qds.fetch import main as fetch_main

    with pytest.raises(SystemExit) as raised:
        fetch_main([])
    assert raised.value.code == 2
    assert "--status" in capsys.readouterr().err


def test_reading_the_catalogue_does_not_import_mflux_or_torch(tmp_path):
    """`qds fetch --status` must not pay for the inference stack to list models.

    In a subprocess, because the assertion is about `sys.modules` and this
    suite's own imports have already populated the one in this process — which is
    how the in-process version of this check ended up written as
    `assert "mflux" not in sys.modules or True`, i.e. as nothing at all.

    The property is load-bearing: the menubar app reads this catalogue on every
    refresh, and model management is supposed to work while the server is down.
    """
    import subprocess
    import sys

    config = tmp_path / "server-config.json"
    config.write_text("{}", encoding="utf-8")

    probe = (
        "import contextlib, io, sys\n"
        "from qds.cli import main\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    rc = main(['fetch', '--status'])\n"
        "heavy = sorted(m for m in ('mflux', 'torch', 'mlx') if m in sys.modules)\n"
        "print(rc, heavy)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "QDS_SERVER_CONFIG": str(config),
            "HF_HOME": str(tmp_path / "hf"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0 []", completed.stdout


# ── The environment contract ───────────────────────────────────────────────


def test_the_canonical_prefix_is_read(monkeypatch):
    monkeypatch.setenv("QDS_SERVER_PORT", "9111")
    monkeypatch.delenv("MFLUX_SERVER_PORT", raising=False)
    assert env.get("PORT") == "9111"


def test_the_retired_prefix_is_still_read(monkeypatch):
    monkeypatch.delenv("QDS_SERVER_PORT", raising=False)
    monkeypatch.setenv("MFLUX_SERVER_PORT", "9112")
    assert env.get("PORT") == "9112"


def test_the_canonical_prefix_wins_over_the_retired_one(monkeypatch):
    monkeypatch.setenv("QDS_SERVER_PORT", "9113")
    monkeypatch.setenv("MFLUX_SERVER_PORT", "9114")
    assert env.get("PORT") == "9113"


def test_the_retired_prefix_is_complained_about(monkeypatch, caplog):
    monkeypatch.delenv("QDS_SERVER_IMAGE_STORE", raising=False)
    monkeypatch.setenv("MFLUX_SERVER_IMAGE_STORE", "/tmp/images")
    monkeypatch.setattr(env, "_warned", set())

    with caplog.at_level("WARNING", logger="qds.env"):
        assert env.get("IMAGE_STORE") == "/tmp/images"

    assert "MFLUX_SERVER_IMAGE_STORE" in caplog.text
    assert "QDS_SERVER_IMAGE_STORE" in caplog.text


def test_an_unset_variable_is_not_an_empty_one(monkeypatch):
    """`""` means "never" for several settings; it must survive as itself."""
    monkeypatch.setenv("QDS_SERVER_LOG_FILE", "")
    assert env.get("LOG_FILE") == ""
    monkeypatch.delenv("QDS_SERVER_LOG_FILE")
    monkeypatch.delenv("MFLUX_SERVER_LOG_FILE", raising=False)
    assert env.get("LOG_FILE") is None
    assert env.get("LOG_FILE", "fallback") == "fallback"


# ── What the settings layer does with it ───────────────────────────────────


def test_the_retired_prefix_still_points_the_server_at_a_config(monkeypatch, tmp_path):
    """The failure this fallback exists to prevent, stated directly."""
    from qds.settings import config_path, load_settings

    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"server": {"port": 9222}}), encoding="utf-8")
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))

    assert config_path() == config
    assert load_settings(strict=False).server.port == 9222


def test_an_empty_config_variable_does_not_fall_back_to_the_packaged_default(monkeypatch):
    """The silent-defaults trap, reached through the *empty* value rather than a stale one.

    `QDS_SERVER_CONFIG=$SOMETHING` in a launch agent, with `SOMETHING` unset,
    hands this process an empty string. Treating that as "unset" would start the
    server on the configuration inside `site-packages`, which is the exact
    failure the variable exists to prevent — and it would do it silently.
    Resolving to `.` instead fails loudly, which is what it did before.
    """
    from qds.settings import DEFAULT_CONFIG_PATH, config_path

    monkeypatch.setenv("QDS_SERVER_CONFIG", "")
    assert config_path() != DEFAULT_CONFIG_PATH
    assert str(config_path()) == "."

    monkeypatch.delenv("QDS_SERVER_CONFIG")
    assert config_path() == DEFAULT_CONFIG_PATH


def test_the_canonical_prefix_overrides_a_setting_from_the_file(monkeypatch, tmp_path):
    from qds.settings import load_settings

    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"server": {"port": 9223}}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    monkeypatch.setenv("QDS_SERVER_PORT", "9224")

    assert load_settings(strict=False).server.port == 9224
