"""JSON Lines logging, and robustness of the write paths.

These tests cover what breaks once the server is launched from a `.app` rather
than from a terminal: paths relative to the current directory, missing parent
directories, and an output format a supervisor can consume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from qds.logs import JsonFormatter, setup_logging
from qds.settings import ServerSettings


def _record(message: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="qds",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_one_line_is_one_valid_json_object():
    line = JsonFormatter().format(_record("step 3/9"))
    payload = json.loads(line)
    assert "\n" not in line
    assert payload["message"] == "step 3/9"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "qds"
    assert "ts" in payload


def test_non_ascii_is_not_escaped():
    """`ensure_ascii=False`, so log text stays readable in a terminal and in the app.

    The probe is deliberately non-ASCII, and deliberately a real message: the
    engine prefixes generation logs with `▶` and separates fields with an em
    dash, and prompts carry accents of their own. Swapping the probe for plain
    ASCII would make the assertion true by construction — the test would stay
    green while covering nothing at all.
    """
    formatted = JsonFormatter().format(_record("▶ z-image-turbo — step 3/9"))
    assert "▶" in formatted
    assert "—" in formatted
    assert "\\u" not in formatted


def test_structured_fields_are_exposed():
    line = JsonFormatter().format(
        _record("step 3/9", event="generation_step", fields={"step": 3, "total": 9})
    )
    payload = json.loads(line)
    assert payload["event"] == "generation_step"
    assert payload["fields"] == {"step": 3, "total": 9}


def test_no_stray_keys_without_extra():
    payload = json.loads(JsonFormatter().format(_record()))
    assert "event" not in payload
    assert "fields" not in payload


def test_unserializable_objects_do_not_crash():
    # `memory_stats()` returns floats, but a badly built `extra` must never break
    # a generation.
    payload = json.loads(JsonFormatter().format(_record(event="x", fields={"path": Path("/tmp/a")})))
    assert payload["fields"]["path"] == "/tmp/a"


def test_json_mode_writes_to_stdout_and_text_mode_to_stderr():
    """Channel separation, and it is not cosmetic.

    mflux renders its denoising bar with tqdm on stderr, as fragments terminated
    by `\\r` with no newline. Our JSON objects would end up glued to them on the
    same segment (`\\r 0%| | 0/40 [...]{"ts": ...}`) and any reasonable consumer
    would miss all of them. Verified for real: 9 lines on stdout, 0 non-JSON,
    tqdm still on stderr.
    """
    import sys

    setup_logging("INFO", None, json_lines=True)
    streams = [h.stream for h in logging.getLogger("qds").handlers]
    assert streams == [sys.stdout]

    setup_logging("INFO", None, json_lines=False)
    streams = [h.stream for h in logging.getLogger("qds").handlers]
    assert streams == [sys.stderr]


def test_log_file_directory_is_created(tmp_path):
    """`FileHandler` neither creates parent directories nor expands `~`."""
    log_file = tmp_path / "logs" / "nested" / "mflux.log"
    setup_logging("INFO", log_file, json_lines=True)
    logging.getLogger("qds").info("hello", extra={"event": "test"})

    for handler in logging.getLogger("qds").handlers:
        handler.flush()
    assert log_file.exists()
    payload = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["event"] == "test"

    # Do not leave a handler open on tmp_path for the following tests.
    setup_logging("INFO", None)


# ── Absolute paths ─────────────────────────────────────────────────────────


def test_write_paths_are_made_absolute():
    """Otherwise, launched from a `.app`, the CWD is `/` and startup fails."""
    settings = ServerSettings.model_validate({"image_store": "images", "log_file": "mflux.log"})
    assert Path(settings.image_store).is_absolute()
    assert Path(settings.log_file).is_absolute()


def test_tilde_is_expanded():
    settings = ServerSettings.model_validate({"log_file": "~/mflux-probe.log"})
    assert "~" not in settings.log_file
    assert settings.log_file.startswith(str(Path.home()))


def test_disabling_the_log_file_stays_possible():
    # The empty string is the documented way to disable the file through
    # QDS_SERVER_LOG_FILE; it must not turn into a path.
    assert ServerSettings.model_validate({"log_file": ""}).log_file == ""
    assert ServerSettings.model_validate({"log_file": None}).log_file is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("log_json", True), ("shutdown_grace_s", 5.0)],
)
def test_new_fields_exist(field, value):
    settings = ServerSettings.model_validate({field: value})
    assert getattr(settings, field) == value


def test_shutdown_grace_must_be_positive():
    with pytest.raises(ValueError):
        ServerSettings.model_validate({"shutdown_grace_s": 0})
