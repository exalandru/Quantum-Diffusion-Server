"""Loading and validation of `server-config.json`.

Every key in the `server` section can be overridden by a `QDS_SERVER_<KEY>`
environment variable, which makes it possible to deploy the same config with a
different binding and API key.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qds import env
from qds.env import ENV_PREFIX
from qds.registry import QUANTIZE_CHOICES, ModelSpec, build_registry

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "server-config.json"
logger = logging.getLogger("qds.settings")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
#: "raw" is a local extension: the PNG bytes straight in the response body.
RESPONSE_FORMATS = {"url", "b64_json", "raw"}

#: When a plane's gate applies. `network` binds it only while the server is
#: reachable beyond this machine; `always` binds it on loopback too.
#:
#: A tightening knob only: there is no value here that opens something the
#: server closes today. Off-loopback exposure keeps its own floor further down
#: in `runtime_issues` — an admin password *and* an api_key — which `network`
#: cannot lower and `always` only adds to.
AuthScope = Literal["network", "always"]


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    #: When set, required as `Authorization: Bearer <key>`.
    api_key: str | None = None
    #: When each plane's gate applies, decided per plane rather than once for
    #: the server. The pair exists because the sensible desktop posture is
    #: asymmetric: the control plane edits the configuration, reads the logs and
    #: restarts the process, so it is worth a password even for someone sitting
    #: at the machine; the playground is the thing that machine's owner uses all
    #: day, and asking them for a password to generate an image locally is
    #: ceremony. A single global scope cannot express `admin: always` with
    #: `playground: network`, which is exactly the posture asked for.
    #:
    #: Flat fields rather than a nested `auth:` block: the two credentials they
    #: govern — `api_key` here, the admin password beside the configuration —
    #: already live at this level, and half the auth story in another place is
    #: how the two halves drift.
    admin_auth_scope: AuthScope = "network"
    playground_auth_scope: AuthScope = "network"
    #: Origins a *browser page* may read a response from. Empty is the default and
    #: means none: `/v1` is open on a keyless loopback install, and a wildcard there
    #: lets any page in any tab spend this machine's GPU and read what came back.
    #: The dashboard and the playground are same-origin, so they need no entry.
    #: Pair a wildcard with an `api_key`, or name the origins that need one.
    cors_origins: list[str] = Field(default_factory=list)
    #: `Host` headers this server answers to, beyond loopback and its own
    #: addresses. Empty derives them; a non-empty list is an allowlist, not an
    #: addition, so it is the way to permit a name no derivation can guess — a
    #: router alias, a DNS entry, a reverse proxy.
    allowed_hosts: list[str] = Field(default_factory=list)
    #: Bounds the OpenAI `n` parameter; images are generated one at a time.
    max_n: int = Field(default=4, ge=1, le=32)
    #: Deadline past which the denoising loop is interrupted.
    request_timeout_s: float = Field(default=900.0, gt=0)
    #: Directory serving images for `response_format="url"`.
    image_store: str = "images"
    #: How long an image served as a `url` survives. `0` keeps them forever, which
    #: is the default: a generated image the user has not saved yet is worth more
    #: than the disk it costs, and the playground's own images already live outside
    #: this directory.
    image_ttl_s: int = Field(default=0, ge=0)
    #: Directory holding the playground's sessions database and its images.
    #: Separate from `image_store`: playground images belong to a durable session
    #: record and must survive the TTL purge.
    #:
    #: `None` means "beside the configuration file", which is where the rest of
    #: this installation's state already lives (`admin-credential.json`). It is
    #: **not** a CWD-relative default: a field default skips validators, so the
    #: relative string that one would be reaches `mkdir` unresolved and lands
    #: wherever the process happens to be — `/` when launched from an app bundle,
    #: which is read-only. See `playground_directory`.
    playground_store: str | None = None
    #: Value used when the client sends no `response_format`.
    #: "url" is the OpenAI default: changing it breaks the SDKs, which read
    #: `data[0].url` and find `None`.
    default_response_format: str = "url"
    max_upload_mb: float = Field(default=25.0, gt=0)
    log_level: str = "INFO"
    log_file: str | None = "mflux.log"
    #: One line, one JSON object. Aimed at a supervisor (the desktop app)
    #: rather than a human in front of a terminal.
    log_json: bool = False
    #: One progress log every N denoising steps (0 = none).
    progress_log_every: int = Field(default=1, ge=0)
    #: Bounds uvicorn's graceful shutdown, whose default is to wait forever on
    #: in-flight connections — which here means up to `request_timeout_s`.
    shutdown_grace_s: float = Field(default=10.0, gt=0)
    #: Release the warm model after this many seconds without a generation.
    #: `None` never releases — the historical behaviour. `0` releases as soon as
    #: the request ends. Meant for sharing unified memory with something else, a
    #: text LLM typically; the cost is paying the load again on the next image.
    idle_unload_s: float | None = Field(default=None, ge=0)

    @field_validator("image_store", "playground_store", "log_file")
    @classmethod
    def _absolute_path(cls, value: str | None) -> str | None:
        """Anchor the write paths, which are CWD-relative by default.

        `image_store` and `log_file` are created during `create_app`, before the
        server even binds. Launched from a `.app` (CWD = `/`, read-only), a
        relative path makes startup fail; launched from anywhere else, it
        scatters `images/` and `mflux.log` into the current directory. So we
        resolve them at validation time.
        """
        if value is None or value == "":
            return value
        return str(Path(value).expanduser().resolve())

    @field_validator("default_response_format")
    @classmethod
    def _check_response_format(cls, value: str) -> str:
        if value not in RESPONSE_FORMATS:
            raise ValueError(f"default_response_format must be one of {sorted(RESPONSE_FORMATS)}")
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log_level: {value!r}")
        return level

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def gate_binds(self, scope: AuthScope) -> bool:
        """Whether a gate with this scope applies to *this* server's binding.

        One truth table, asked twice, so the two planes cannot answer the same
        question differently: `always` binds everywhere, `network` binds only
        once the socket is reachable beyond this machine.
        """
        return scope == "always" or not self.is_loopback

    @property
    def admin_gate_binds(self) -> bool:
        return self.gate_binds(self.admin_auth_scope)

    @property
    def playground_gate_binds(self) -> bool:
        return self.gate_binds(self.playground_auth_scope)


#: Where `huggingface_hub` keeps its cache when nothing says otherwise. Its own
#: default, and the one every QDS install has used so far.
DEFAULT_HF_HOME = "~/.cache/huggingface"


class StorageSettings(BaseModel):
    """Where large, long-lived files live: what is downloaded, and what is made.

    Separate from `server` on purpose: this is about the machine's disks, not
    about the HTTP surface. The two roots here are independent — either can sit
    on an external volume without the other following it.
    """

    #: Root for downloaded weights, i.e. `HF_HOME`. `None` keeps
    #: `huggingface_hub`'s own default, which is what every install used before
    #: this setting existed. An absolute path is required: this is resolved by
    #: processes whose working directory is `/` when launched from Finder.
    hf_home: str | None = None
    #: Root for what QDS *generates* — pre-quantized copies, their completion
    #: markers, component progress. `None` derives the application's own data
    #: directory (`artifacts.default_cache_root`).
    #:
    #: Independent of `hf_home` on purpose, and not merely as a convenience: one
    #: holds weights that can be downloaded again and the other holds hours of
    #: conversion that cannot, so they are the two directories a user is most
    #: likely to want on different disks.
    cache_dir: str | None = None

    @field_validator("hf_home", "cache_dir")
    @classmethod
    def _absolute_storage_path(cls, value: str | None, info) -> str | None:
        if value is None or value.strip() == "":
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"storage.{info.field_name} must be an absolute path (got {value!r}). A relative "
                f"path would resolve against the working directory, which is '/' for an app "
                f"launched from Finder."
            )
        # Deliberately not `.resolve()`: that walks symlinks and would fail on a
        # path whose external volume is not mounted, which is a state this app
        # must be able to hold and report rather than reject.
        return str(path)


class RewriteSettings(BaseModel):
    """Prompt rewriting: whether it is offered, and how it is bounded.

    **On** by default, which it was not at first, and the reason it changed is
    worth recording because the original reason was good.

    It shipped `false` because the first Enhance fetched a gigabyte of weights,
    and a server nobody had asked for it should not advertise a control that
    would start that download. Two things dissolved that: the menubar app now
    fetches the weights at install time (`qds fetch --rewriter`), so on an app
    install they are already there; and independently, the consent surface that
    was missing now exists -- `/v1/capabilities` publishes `downloaded`, and the
    composer says "First use downloads N MB" before anything is pressed. The
    flag was standing in for a warning that now exists.

    A `pip install qds` user still pays that download on first use, and is told
    so before paying it. Setting `enabled: false` remains the way to make the
    control disappear entirely.

    The decoder itself (`mlx-lm`) is an ordinary runtime dependency and arrives
    with the server -- it was an optional extra first, which made this
    docstring's reasoning read better and left the app's installer unable to
    satisfy it. With `enabled: false` the route refuses `rewrite=on` with a 409,
    `/v1/capabilities` publishes `available: false` with the reason, and the
    dashboard hides the control — one switch, three consistent answers.

    None of these knobs can move the engine's memory bound. That lives in
    `qds/rewrite/catalogue.py`, is enforced at import, and is deliberately not
    configurable: raising it is a decision about `ModelEngine`'s third slot, not
    a preference.
    """

    enabled: bool = True
    #: Catalogue key from `qds.rewrite.catalogue`. Named `model` for symmetry
    #: with `default_model`, though it is a different catalogue -- which is why
    #: the three key namespaces are asserted disjoint in `tests/test_rewrite.py`.
    model: str = "qwen3-4b-2507-4bit"
    #: Prompts of this many words or more are generated as typed, without
    #: calling the rewriter at all.
    #:
    #: This is the mechanism that keeps a carefully written prompt intact, and
    #: it is here rather than in the system prompt because the model cannot be
    #: trusted with it: asked to leave long prompts unchanged, the shipped
    #: rewriter obeyed 8 times in 18 *and* got measurably worse at everything
    #: else, the rule having competed for a small model's attention.
    #:
    #: 40 is where the evaluation set separates: hand-written detailed prompts
    #: ran 40-45 words, while an expanded short prompt lands at 83. A user who
    #: wants the boundary elsewhere moves it; a user who wants it never to apply
    #: sets it very high and accepts what the model does with their paragraph.
    word_ceiling: int = 40
    #: Longest rewrite to decode. Bounded above by the catalogue's
    #: `MAX_NEW_TOKENS`, which is what the KV-cache argument is computed from,
    #: so this can only ever be lowered from it.
    max_new_tokens: int = 320
    #: Sampling temperature. 0.7 is measured; lower is not obviously better --
    #: at 0.3 the shipped model produced the same structures with less variety,
    #: and a rewrite is regenerated by asking again.
    temperature: float = 0.7
    #: Wall-clock bound on one decode, checked between tokens.
    #:
    #: Re-derived after the output target tripled and the model changed:
    #: measured p95 is 2.35 s for load, decode and unload together, against
    #: 0.62 s at the previous 83-word median. 30 s is still comfortably over the
    #: 3x-p95 this bound is held to. It exists to end a runaway decode, not to
    #: shape latency, and it is the one knob here with no catalogue ceiling.
    timeout_s: float = 30.0
    #: Replaces `prompt.DEFAULT_SYSTEM_PROMPT` when set. A quality knob, so it
    #: is configurable -- unlike the bounds above it, which are safety knobs and
    #: are not.
    system_prompt: str | None = None

    @field_validator("word_ceiling", "max_new_tokens")
    @classmethod
    def _positive(cls, value: int, info) -> int:
        if value < 1:
            raise ValueError(f"rewrite.{info.field_name} must be at least 1 (got {value}).")
        return value

    @field_validator("word_ceiling")
    @classmethod
    def _below_the_prompt_bound(cls, value: int) -> int:
        # A quality knob, but not an unbounded one: a ceiling at or past
        # `MAX_PROMPT_TOKENS` would send prompts to a decode that refuses them,
        # turning "generated as typed" into a recorded rewrite failure for every
        # long prompt. Strictly below, because a word is worth more than a
        # token: even in English the ceiling must leave room for the system
        # prompt, which is in the templated text the bound is measured on.
        from qds.rewrite.catalogue import MAX_PROMPT_TOKENS

        if value >= MAX_PROMPT_TOKENS:
            raise ValueError(
                f"rewrite.word_ceiling is {value}, at or over the "
                f"{MAX_PROMPT_TOKENS}-token bound the rewriter's decode enforces. "
                "Words are not tokens: the ceiling must stay below that bound "
                "rather than meet it."
            )
        return value

    @field_validator("max_new_tokens")
    @classmethod
    def _within_the_engine_bound(cls, value: int) -> int:
        from qds.rewrite.catalogue import MAX_NEW_TOKENS

        if value > MAX_NEW_TOKENS:
            raise ValueError(
                f"rewrite.max_new_tokens is {value}, over the {MAX_NEW_TOKENS} the "
                "engine's KV-cache bound is computed from. Raising it is a "
                "decision about `ModelEngine`'s third slot, not a setting."
            )
        return value


class MCPSettings(BaseModel):
    """The MCP surface: whether it is offered, and how far it may reach.

    MCP is the third plane. `/v1` serves other *applications*, `/playground/api`
    serves a *person* at a browser, and `/mcp` serves a *model* — which is the
    distinction every setting here exists for. A model is not a hostile party,
    but it is an untrusted one: what it asks for may have been written into a
    prompt by someone else, so the arguments it chooses get bounds the other two
    planes do not need.

    On by default, matching `RewriteSettings`: the SDK arrives with the server,
    nothing is downloaded on first use, and a surface that must be discovered in
    a config file is a surface nobody finds. Setting `enabled: false` removes
    the route entirely rather than answering it with a refusal.

    There is deliberately no `mcp.max_n`. `server.max_n` is the one authority on
    how many images a request may ask for, and `check_n` already enforces it on
    every plane. A second ceiling would be two numbers for one rule, and the
    lower one would silently win.

    Nothing here is reachable through `QDS_SERVER_*`: `_env_overrides` covers
    `ServerSettings` only, as it does for `rewrite`. This is configuration for
    an installation, set in the file or through the dashboard.
    """

    enabled: bool = True
    #: How long a generation tool waits before returning the generation id
    #: instead of the images. Not a failure when it elapses -- the work is
    #: queued and durable, and `wait_for_generation` resumes it.
    #:
    #: Generous because the alternative is worse: a client that gets an id back
    #: has to be told to call another tool, and a small model asked to do that
    #: mid-conversation frequently does not. Ten minutes covers a cold model
    #: load plus an n=4 run on the shipped default.
    tool_timeout_s: float = Field(default=600.0, gt=0)
    #: How often that wait re-reads the generation row. Reading a row, not the
    #: engine: the row is authoritative for *this* job, and the engine's
    #: snapshot is not (see `qds/mcp/progress.py`).
    poll_interval_s: float = Field(default=0.5, gt=0)
    #: Directories a model-chosen filesystem path may name. Empty by default,
    #: and that is the setting's whole point: `reference_path` is an argument
    #: the *model* fills in, so an unbounded one would let a prompt-injected
    #: model publish any readable file into the playground store -- which is
    #: served over HTTP behind a credential a loopback install does not have.
    #: Fail closed, and let an operator open exactly the directory they meant.
    image_roots: list[str] = Field(default_factory=list)
    #: Whether `delete_image` and `delete_group` are offered at all. Off by
    #: default: deletion in the playground is a click a human makes with the
    #: image in front of them, and a tool has no equivalent of that confirmation.
    allow_destructive: bool = False

    @field_validator("image_roots")
    @classmethod
    def _absolute_roots(cls, value: list[str]) -> list[str]:
        roots: list[str] = []
        for entry in value:
            if entry is None or entry.strip() == "":
                continue
            path = Path(entry).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    f"mcp.image_roots entries must be absolute paths (got {entry!r}). A "
                    f"relative path would resolve against the working directory, which is "
                    f"'/' for an app launched from Finder -- and this setting is a "
                    f"containment boundary, so a root that moves with the CWD contains "
                    f"nothing."
                )
            roots.append(str(path))
        return roots


class ModelOverride(BaseModel):
    # `model_path` falls into pydantic's reserved `model_` namespace; we free it
    # up rather than rename a field that speaks to mflux.
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    quantize: int | None = None
    #: Local path or HF repo substituted for the catalogue's. Required for
    #: `flux2-dev`, whose pre-quantized artifact is machine-specific.
    model_path: str | None = None
    default_size: str | None = None
    default_steps: int | None = Field(default=None, ge=1)
    default_guidance: float | None = Field(default=None, ge=0)
    enable_edit: bool | None = None
    #: Which saved, already-quantized variant of *this source* to generate with.
    #: `None` uses the source itself. A bit depth selects the validated artifact
    #: for the current `model_path`/repo at that precision — deliberately a
    #: separate key from `model_path`, which stays the source's identity, so
    #: activating a conversion never rewrites what the model *is*.
    prequantized_variant: int | None = None
    #: Sampler preset, for the models that have any. `ideogram-4` only, whose step
    #: count and guidance schedule come as a named bundle.
    preset: str | None = None

    @field_validator("quantize")
    @classmethod
    def _check_quantize(cls, value: int | None) -> int | None:
        if value is not None and value != 0 and value not in QUANTIZE_CHOICES:
            raise ValueError(f"quantize must be 0 (none) or one of {list(QUANTIZE_CHOICES)}")
        return value


@dataclass(frozen=True)
class RuntimeIssue:
    """One broken runtime invariant, as data rather than as an exception.

    Structured because both consumers need different things from it: the server
    raises it as a message, and the catalogue publishes it for an interface that
    has to say what to do about it. A traceback served neither.
    """

    code: str
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message}


class ConfigError(ValueError):
    """An expected configuration failure, carrying a code for the interface.

    A `ValueError` subclass so every existing caller — `run_guarded`, the CLI
    entry points, the tests that assert on invalid configuration — keeps working
    unchanged.
    """

    def __init__(self, message: str, *, code: str = "invalid_config", field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    #: `z-image-turbo` because a default has to work with nothing set up: its repo
    #: is neither gated nor licence-restricted (Apache-2.0), it needs no
    #: preparation step, 9 steps make it the fastest usable model here, and its
    #: native 1280x720 matches the resolution the shipped config asks for.
    default_model: str = "z-image-turbo"
    #: Config-wide generation resolution, `"WxH"`. Sits below the per-model
    #: `default_size` overrides and above the catalogue, so one knob covers every
    #: model while a single model can still be pinned. `null` keeps each model on
    #: its catalogue size. Lives here rather than under `server` because it is a
    #: generation default, not a transport setting — same level as `default_model`.
    default_size: str | None = None
    # There is deliberately no config-wide `default_quantize`. It existed, and it
    # overwrote the catalogue rather than standing behind it, so a single global
    # silently decided the precision of every model — including the ones whose
    # catalogue row had picked one on purpose. Precision is a property of a model,
    # not a property of an installation: a 2B is unusable at 4-bit where a 20B is
    # fine. `models.<key>.quantize` remains, for the one model you mean.
    #
    # A config that still carries the key is not rejected — `load_settings` says
    # out loud that it is ignored, rather than letting pydantic drop it in
    # silence, which is how a removed setting turns into a mystery.
    models: dict[str, ModelOverride] = Field(default_factory=dict)

    def runtime_issues(self) -> list[RuntimeIssue]:
        """Invariants a *generation server* must satisfy before it may serve.

        Deliberately not `@model_validator`, and that distinction is the whole
        point of this method. Every check below is about serving generations:
        which model answers a request that names none, and whether a socket open
        to the network is authenticated. None of them is about *scanning
        sources* — which repositories are cached, how big they are, where a
        `model_path` points — and all of them used to be enforced at
        construction, so `Settings(...)` raised and every reader died with it.

        The reader that suffered was model management. `qds fetch
        --status` builds the catalogue, and a configuration whose
        `default_model` had been switched off made the whole catalogue
        unreadable: a traceback instead of a list, and no way to reach the
        controls that would have repaired it. The invariant was real and the
        blast radius was wrong.

        So the checks live here, returned as data. `load_settings(strict=True)`
        — the server's path — raises on the first one. `strict=False` — the
        catalogue's path — carries them alongside the rows as warnings the
        interface can act on. Nothing is repaired here, silently or otherwise:
        a broken invariant stays broken until the user fixes it.
        """
        issues: list[RuntimeIssue] = []
        # The effective catalogue, not the built-in one: an imported model is a
        # perfectly good default, and validating against `BASE_SPECS` alone made
        # it impossible to choose one. Membership decides — never the shape of
        # the key.
        effective = self.registry(include_disabled=True)
        if self.default_model not in effective:
            issues.append(
                RuntimeIssue(
                    code="unknown_default_model",
                    field="default_model",
                    message=(
                        f"Unknown default_model: {self.default_model!r}. Valid keys: "
                        f"{sorted(effective)}. An imported model must still be registered; "
                        f"if it was forgotten, choose another."
                    ),
                )
            )
        else:
            override = self.models.get(self.default_model)
            if override is not None and not override.enabled:
                issues.append(
                    RuntimeIssue(
                        code="default_model_disabled",
                        field="default_model",
                        message=(
                            f'Default model "{self.default_model}" is disabled. Enable it or '
                            f"choose another default model."
                        ),
                    )
                )
        if not self.server.is_loopback and not credential_is_set():
            issues.append(
                RuntimeIssue(
                    code="unauthenticated_admin",
                    field="server.admin_password",
                    message=(
                        f"host={self.server.host!r} exposes this server beyond this machine: "
                        f"an admin password is mandatory before the control plane can be "
                        f"reachable from the network. Set one in the dashboard."
                    ),
                )
            )
        if not self.server.is_loopback and not self.server.api_key:
            issues.append(
                RuntimeIssue(
                    code="unauthenticated_host",
                    field="server.api_key",
                    message=(
                        f"host={self.server.host!r} exposes the server beyond this machine: "
                        f"an api_key is mandatory (config or {ENV_PREFIX}API_KEY)."
                    ),
                )
            )
        # A gate that is on but has no key is a gate that is off, so each plane
        # is checked against *its own* credential and names *its own* field: a
        # misconfigured playground reported as an admin problem sends the person
        # repairing it to the wrong screen. Fail closed, the way an off-loopback
        # server without an admin password already does — the alternative is a
        # scope that silently means nothing.
        if self.server.admin_auth_scope == "always" and not credential_is_set():
            issues.append(
                RuntimeIssue(
                    code="admin_password_required_by_scope",
                    field="server.admin_auth_scope",
                    message=(
                        "admin_auth_scope='always' asks for the admin password even on this "
                        "machine, and no admin password is set. Set one in the dashboard, or "
                        "put admin_auth_scope back to 'network'."
                    ),
                )
            )
        if self.server.playground_auth_scope == "always" and not playground_credential_is_set():
            issues.append(
                RuntimeIssue(
                    code="playground_password_required_by_scope",
                    field="server.playground_auth_scope",
                    message=(
                        "playground_auth_scope='always' asks for the playground password even "
                        "on this machine, and no playground password is set. Set one in the "
                        "dashboard, or put playground_auth_scope back to 'network'."
                    ),
                )
            )
        return issues

    @property
    def effective_hf_home(self) -> str:
        """The HuggingFace root actually in force, resolved in one place.

        Precedence: the configuration, then an inherited `HF_HOME`, then
        huggingface_hub's default. Configuration wins over the environment
        because it is the setting the user chose in the app, whereas the
        environment is whatever the launcher happened to carry.
        """
        if self.storage.hf_home:
            return self.storage.hf_home
        inherited = os.environ.get("HF_HOME")
        if inherited:
            return inherited
        return str(Path(DEFAULT_HF_HOME).expanduser())

    @property
    def effective_hub_cache(self) -> str:
        """Where the cached repositories actually are, inside the storage root.

        `HF_HOME` and the hub cache are two different directories, and the
        setting names the first. Deriving the second by appending `hub`
        unconditionally is what made a storage folder that *is* a hub cache
        report every model as missing — see `availability.hub_cache_for`.
        """
        from qds.availability import hub_cache_for

        return str(hub_cache_for(self.effective_hf_home))

    @property
    def effective_cache_dir(self) -> str:
        """Where generated artifacts are read from and written to.

        The configured directory, or the application's own. Changing the setting
        changes where *future* artifacts are created and where discovery looks;
        nothing is moved, copied or deleted, because tens of gigabytes are not
        something to relocate as a side effect of a form field.
        """
        from qds import artifacts

        return self.storage.cache_dir or str(artifacts.default_cache_root())

    def apply_hf_home(self) -> str:
        """Publish the effective root into the environment, and return it.

        This has to happen before anything imports `huggingface_hub`, which
        freezes `HF_HUB_CACHE` at import time — so it belongs at the top of each
        entry point rather than at the point of use. Setting both variables keeps
        a stale inherited `HF_HUB_CACHE` from silently winning over the root the
        user configured, and makes every consumer — this app, mflux and
        huggingface_hub alike — agree on which directory holds the weights.
        """
        root = self.effective_hf_home
        os.environ["HF_HOME"] = root
        os.environ["HF_HUB_CACHE"] = self.effective_hub_cache
        return root

    def rewriter(self) -> Any:
        """The configured `RewriterSpec`, or `None` if rewriting is unavailable.

        `None` covers both "switched off" and "names a key this build does not
        have" on purpose: from every caller's point of view those are the same
        state -- the feature is not on offer -- and distinguishing them at each
        call site is how one of the two ends up unhandled. The *reason* is not
        lost: `rewrite_unavailable_reason` reports it, and that is what
        `/v1/capabilities` publishes.
        """
        from qds.rewrite.catalogue import by_key

        if not self.rewrite.enabled:
            return None
        return by_key(self.rewrite.model)

    def rewrite_unavailable_reason(self) -> str | None:
        """Why rewriting is not on offer, or `None` if it is.

        Deliberately not raising: this is read to *describe* the server, on a
        path (`/v1/capabilities`) that must answer even when the answer is bad
        news.
        """
        from qds.rewrite.catalogue import KEYS, by_key

        if not self.rewrite.enabled:
            return "Prompt rewriting is switched off (`rewrite.enabled`)."
        if by_key(self.rewrite.model) is None:
            return f"Unknown rewriter {self.rewrite.model!r}. Valid keys: {sorted(KEYS)}."
        return None

    def registry(self, *, include_disabled: bool = False) -> dict[str, ModelSpec]:
        """The configured catalogue.

        `include_disabled` keeps the entries the server will not expose. Downloading
        and reporting need it: the documented workflow is to fetch a model *before*
        turning it on, and dropping disabled entries there silently discarded their
        `model_path` and `quantize` overrides — so the Models tab named one repo and
        the Install button fetched another.
        """
        from qds import library

        try:
            imported = library.load()
        except library.LibraryTooNew:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.warning("Imported-model library unavailable", exc_info=True)
            imported = []
        return build_registry(
            self.models,
            default_size=self.default_size,
            include_disabled=include_disabled,
            imported=imported,
            cache_root=self.effective_cache_dir,
        )


def _coerce_env(raw: str, field_name: str) -> Any:
    if field_name == "cors_origins":
        return [item.strip() for item in raw.split(",") if item.strip()]
    # An empty variable is how the environment says "leave this unset": the
    # desktop app already uses it for `log_file`, and `idle_unload_s` needs it to
    # be able to mean "never" rather than fail validation.
    if raw == "" and field_name in {"api_key", "log_file", "idle_unload_s"}:
        return None
    return raw


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field_name in ServerSettings.model_fields:
        raw = env.get(field_name.upper())
        if raw is not None:
            overrides[field_name] = _coerce_env(raw, field_name)
    return overrides


def config_path() -> Path:
    # `env.get(name, default)`, not `env.get(name) or default`: an *empty*
    # `QDS_SERVER_CONFIG` — what a launch agent produces from an unset shell
    # variable — must stay empty and fail loudly, rather than fall through to the
    # packaged default. Falling through is precisely the silent
    # "running on defaults, nothing in the log to say why" this variable exists
    # to prevent.
    return Path(env.get("CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()


def playground_directory(server: ServerSettings) -> Path:
    """Where the playground keeps its database and its images.

    Anchored to the configuration file when unset, the same way
    `credential.credential_path` anchors `admin-credential.json`: this is state
    that belongs to one installation, and an installation is identified by its
    config file, not by the directory someone launched from.

    The alternative — a CWD-relative default — is what broke the app bundle: a
    field default bypasses `_absolute_path`, so `"playground"` arrived at `mkdir`
    still relative and resolved against `/`, which is read-only. An *explicit*
    value keeps the CWD-relative-then-absolute behaviour of `image_store`, since
    a written-down relative path is a choice rather than an accident.
    """
    if server.playground_store:
        return Path(server.playground_store).expanduser()
    return config_path().parent / "playground"


def configured_default_model(path: Path | None = None) -> str:
    """The `default_model` currently written down, without validating the rest.

    Read raw rather than through `load_settings`, and for a specific reason: the
    caller is `qds import`, deciding whether removing a registration
    would leave this key dangling. Going through validation would make that answer
    depend on the whole config being loadable — and the one config guaranteed to
    fail validation is the one whose `default_model` names a model that no longer
    exists, which is exactly the state this check exists to prevent.

    Same precedence as `load_settings`: the environment override wins, then the
    file, then the field's own default.
    """
    override = env.get("DEFAULT_MODEL")
    if override:
        return override
    path = path or config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # No config yet: the app writes one carrying the field default, so that is
        # what the next start will use.
        raw = {}
    except (OSError, json.JSONDecodeError) as exc:
        # Unreadable is not "unset". Answering with the default here would let a
        # caller conclude "not the default model" about a file it never read.
        raise ValueError(f"{path} could not be read: {exc}") from exc
    value = raw.get("default_model") if isinstance(raw, dict) else None
    if isinstance(value, str) and value:
        return value
    return str(Settings.model_fields["default_model"].default)


def credential_is_set() -> bool:
    """Whether an admin password exists.

    Imported lazily: `qds.credential` reads `config_path()` from this module, and
    a module-level import would be a cycle.
    """
    from qds import credential

    return credential.is_set()


def playground_credential_is_set() -> bool:
    """Whether a playground password exists. Lazy for the same reason."""
    from qds import credential

    return credential.PLAYGROUND.is_set()


#: Set by `load_settings` when no config file was found. `setup_logging` only
#: runs afterwards (in create_app), so we cannot log at the point of discovery —
#: `create_app` takes care of it.
missing_config_path: Path | None = None


def load_settings(path: Path | None = None, *, strict: bool = True) -> Settings:
    """Read `server-config.json`, apply environment overrides, validate.

    A missing config is not an error: the catalogue defaults are enough to
    start. But it is a trap once the package is installed as a wheel, where
    `DEFAULT_CONFIG_PATH` lands in `site-packages/`: we would silently fall back
    to every default. Hence `missing_config_path`, which `create_app` logs as a
    `warning`.

    `strict` selects which of the two contracts this call is asking for. The
    structural one — what the file says, whether it parses, whether each model
    override is well formed — is always enforced, because a caller cannot do
    anything sensible with a document it cannot read. The *runtime* one —
    `runtime_issues`, the invariants a generation server must satisfy before it
    serves — is enforced only when `strict`.

    Model management passes `strict=False` and reports those issues instead. A
    configuration whose default model is switched off is genuinely invalid for
    the server and perfectly readable for the catalogue, and the catalogue is
    where the controls to repair it live: failing there took away the only route
    back to a working configuration.
    """
    global missing_config_path

    path = path or config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        missing_config_path = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object at the root.")
    else:
        missing_config_path = path

    server_raw = dict(raw.get("server") or {})
    server_raw.update(_env_overrides())
    raw["server"] = server_raw

    # The generic `QDS_SERVER_*` loop above only covers `ServerSettings`
    # fields; these two live one level up and are special-cased.
    for field_name in ("default_model", "default_size"):
        value = env.get(field_name.upper())
        if value:
            raw[field_name] = value

    # `default_quantize` was a config-wide bit depth that overwrote every model's
    # catalogue precision. Removed, because it decided for models it knew nothing
    # about. Announced rather than dropped: pydantic ignores unknown keys, so a
    # user who set this would otherwise see their models change precision with
    # nothing said.
    if raw.pop("default_quantize", None) is not None or env.get("DEFAULT_QUANTIZE"):
        logger.warning(
            "default_quantize is no longer supported and was ignored. Precision now "
            "comes from each model's catalogue entry; set models.<key>.quantize to "
            "override one model (0 for bf16)."
        )

    try:
        settings = Settings.model_validate(raw)
        # Surface model-override errors right away rather than on the first
        # request.
        settings.registry()
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration ({path}):\n{exc}") from exc

    if strict:
        for issue in settings.runtime_issues():
            raise ConfigError(
                f"Invalid configuration ({path}): {issue.message}",
                code=issue.code,
                field=issue.field,
            )
    return settings


def recovery_settings() -> Settings:
    """Settings for a server whose configuration file cannot be read at all.

    Not a fallback that hides the problem — the caller has already decided to
    start in recovery mode and will say so on every endpoint. This answers a
    narrower question: *where should that recovery server listen?*

    The environment, then the defaults, and deliberately nothing from the file:
    the file is the thing that could not be parsed. `QDS_SERVER_PORT` matters
    most, because whatever launched this process is waiting on a specific port
    and a recovery server nobody can reach is no better than no server at all.
    """
    try:
        return Settings.model_validate({"server": _env_overrides()})
    except (ValidationError, ValueError):
        # Even the environment is contradictory. Bare defaults still give a
        # loopback port to serve the repair screen on.
        return Settings()
