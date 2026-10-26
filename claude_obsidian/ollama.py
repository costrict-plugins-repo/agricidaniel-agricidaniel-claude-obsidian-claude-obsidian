"""Shared proof that an Ollama model is installed and local-only.

This module also owns the direct JSON transport used by every provider
helper: no ambient proxies, no automatic redirects, one bounded response
read, strict duplicate-rejecting JSON, and one wall-clock deadline per
request. The deadline is enforced on every socket send and receive, so a
server that drips bytes cannot hold a request past its stated timeout.
"""

import functools
import http.client
import io
import re
import time
import urllib.request

from claude_obsidian.json_utils import strict_json_loads


MODEL_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)
DIGEST_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


class OllamaLocalProofError(ValueError):
    """The server response does not prove local-only model execution."""


class OllamaModelUnavailable(OllamaLocalProofError):
    """The requested model or its permitted alias is not installed."""


def _valid_model_name(model):
    if not isinstance(model, str) or MODEL_RE.fullmatch(model) is None:
        raise OllamaLocalProofError("model identity is malformed")
    repository = model.split(":", 1)[0]
    if any(part in {"", ".", ".."} for part in repository.split("/")):
        raise OllamaLocalProofError("model identity is malformed")
    return model


def _cloud_tag(model):
    return ":" in model and "cloud" in model.rsplit(":", 1)[1].lower()


def _remote_identity(value):
    if isinstance(value, dict):
        if any(
            value.get(key) not in (None, "")
            for key in ("remote_host", "remote_model")
        ):
            return True
        return any(_remote_identity(item) for item in value.values())
    if isinstance(value, list):
        return any(_remote_identity(item) for item in value)
    return False


def _valid_digest(value):
    return isinstance(value, str) and DIGEST_RE.fullmatch(value) is not None


def normalize_model_digest(value):
    if not _valid_digest(value):
        raise OllamaLocalProofError("installed model digest is unavailable")
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _aliases(model):
    return (model,) if ":" in model else (model, f"{model}:latest")


def _matching_entries(model, models):
    _valid_model_name(model)
    if _cloud_tag(model):
        raise OllamaLocalProofError("cloud model tags are not local-only")
    if not isinstance(models, list):
        raise OllamaLocalProofError("server model list is malformed")
    aliases = _aliases(model)
    matches = []
    for entry in models:
        if not isinstance(entry, dict):
            raise OllamaLocalProofError("server model entry is malformed")
        name = entry.get("name")
        if name not in aliases:
            continue
        if not isinstance(name, str) or _cloud_tag(name):
            raise OllamaLocalProofError("cloud model tags are not local-only")
        if entry.get("model") not in (None, "", name):
            raise OllamaLocalProofError("installed model identity conflicts")
        if _remote_identity(entry):
            raise OllamaLocalProofError("installed model has remote identity")
        digest = normalize_model_digest(entry.get("digest"))
        matches.append((name, digest))
    if not matches:
        raise OllamaModelUnavailable("requested model is not installed")
    if len({digest for _name, digest in matches}) != 1:
        raise OllamaLocalProofError("installed model aliases have conflicting digests")
    preferred = next((item for item in matches if item[0] == model), matches[0])
    return preferred


def model_is_available(model, models):
    """Return whether tags contain one exact permitted identity with valid proof."""
    try:
        _matching_entries(model, models)
    except OllamaLocalProofError:
        return False
    return True


def prove_local_model(model, get_json, post_json, *, require_explicit_tag=False):
    """Return resolved model identity and digest after local-only preflight."""
    _valid_model_name(model)
    if require_explicit_tag and ":" not in model:
        raise OllamaLocalProofError("an explicit model tag is required")
    status = get_json("status")
    if (
        not isinstance(status, dict)
        or _remote_identity(status)
        or not isinstance(status.get("cloud"), dict)
        or status["cloud"].get("disabled") is not True
    ):
        raise OllamaLocalProofError("server does not prove cloud is disabled")
    tags = get_json("tags")
    if not isinstance(tags, dict):
        raise OllamaLocalProofError("server model list is malformed")
    # Other installed models may be cloud aliases; they never receive this
    # request. Remote identity is decisive only for the list envelope itself
    # and for the entries matching the requested model (_matching_entries).
    if _remote_identity({key: value for key, value in tags.items() if key != "models"}):
        raise OllamaLocalProofError("server model list reports remote identity")
    resolved_model, digest = _matching_entries(model, tags.get("models"))
    shown = post_json("show", {"model": resolved_model})
    if not isinstance(shown, dict) or _remote_identity(shown):
        raise OllamaLocalProofError("model metadata does not prove a local model")
    model_info = shown.get("model_info")
    if not isinstance(model_info, dict) or not model_info:
        raise OllamaLocalProofError("model metadata does not prove a local model")
    for key in ("model", "name"):
        if shown.get(key) not in (None, "", resolved_model):
            raise OllamaLocalProofError("model metadata identity conflicts")
    shown_digest = shown.get("digest")
    if shown_digest not in (None, ""):
        if normalize_model_digest(shown_digest) != digest:
            raise OllamaLocalProofError("model metadata digest conflicts")
    return {"model": resolved_model, "digest": digest}


def validate_model_response(response, proof):
    """Reject remote markers or a model identity that conflicts with proof."""
    if not isinstance(response, dict) or _remote_identity(response):
        raise OllamaLocalProofError("provider response has remote identity")
    model = proof.get("model") if isinstance(proof, dict) else None
    raw_digest = proof.get("digest") if isinstance(proof, dict) else None
    if not isinstance(model, str) or not _valid_digest(raw_digest):
        raise OllamaLocalProofError("verified model identity is malformed")
    digest = normalize_model_digest(raw_digest)
    for key in ("model", "name"):
        if response.get(key) not in (None, "", model):
            raise OllamaLocalProofError("provider response model identity conflicts")
    returned_digest = response.get("digest")
    if returned_digest not in (None, ""):
        if normalize_model_digest(returned_digest) != digest:
            raise OllamaLocalProofError("provider response digest conflicts")
    return response


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect; a provider reply must come from the exact URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RequestDeadlineExceeded(TimeoutError):
    """A provider request exceeded its wall-clock deadline."""


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RequestDeadlineExceeded("provider request deadline exceeded")
    return remaining


class _DeadlineReader(io.RawIOBase):
    """Socket reader whose every receive is bounded by the request deadline."""

    def __init__(self, raw, sock, deadline):
        super().__init__()
        self._raw = raw
        self._sock = sock
        self._deadline = deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        self._sock.settimeout(_remaining(self._deadline))
        return self._raw.readinto(buffer)

    def close(self):
        if not self.closed:
            self._raw.close()
        super().close()


class _DeadlineSocket:
    """Delegate a connected socket while applying one deadline to all I/O."""

    def __init__(self, sock, deadline):
        self._sock = sock
        self._deadline = deadline

    def makefile(self, mode="r", *_args, **_kwargs):
        if mode != "rb":
            raise ValueError("deadline sockets support binary response reads only")
        # The real SocketIO holds the socket's io reference, so the connection
        # may drop its handle while the response body is still being read.
        raw = self._sock.makefile("rb", buffering=0)
        return io.BufferedReader(_DeadlineReader(raw, self._sock, self._deadline))

    def sendall(self, data, *args):
        self._sock.settimeout(_remaining(self._deadline))
        return self._sock.sendall(data, *args)

    def __getattr__(self, name):
        return getattr(self._sock, name)


def _deadline_create_connection(create_connection, deadline):
    """Reset a fresh socket's timeout to what remains after the TCP connect.

    A TLS handshake runs on that socket next, so it must not receive the
    whole timeout again.
    """

    def create(*args, **kwargs):
        sock = create_connection(*args, **kwargs)
        try:
            sock.settimeout(_remaining(deadline))
        except BaseException:
            sock.close()
            raise
        return sock

    return create


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)
        self._create_connection = _deadline_create_connection(
            self._create_connection, deadline
        )

    def connect(self):
        self.timeout = _remaining(self._deadline)
        super().connect()
        self.sock = _DeadlineSocket(self.sock, self._deadline)


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline):
        super().__init__()
        self._deadline = deadline

    def http_open(self, req):
        connection = functools.partial(_DeadlineHTTPConnection, deadline=self._deadline)
        return self.do_open(connection, req)


_DeadlineHTTPSHandler = None
if hasattr(http.client, "HTTPSConnection") and hasattr(urllib.request, "HTTPSHandler"):

    class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
        def __init__(self, *args, deadline, **kwargs):
            self._deadline = deadline
            super().__init__(*args, **kwargs)
            self._create_connection = _deadline_create_connection(
                self._create_connection, deadline
            )

        def connect(self):
            # The TLS handshake uses the time left after the TCP connect as its
            # socket timeout; later sends and receives share the same deadline.
            self.timeout = _remaining(self._deadline)
            super().connect()
            _remaining(self._deadline)
            self.sock = _DeadlineSocket(self.sock, self._deadline)

    class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
        def __init__(self, deadline):
            super().__init__()
            self._deadline = deadline

        def https_open(self, req):
            connection = functools.partial(
                _DeadlineHTTPSConnection, deadline=self._deadline
            )
            kwargs = {"context": self._context}
            check_hostname = getattr(self, "_check_hostname", None)
            if check_hostname is not None:
                kwargs["check_hostname"] = check_hostname
            return self.do_open(connection, req, **kwargs)


def direct_handlers(timeout):
    """Handlers for one request: no proxies, no redirects, one deadline."""
    deadline = time.monotonic() + timeout
    handlers = [
        urllib.request.ProxyHandler({}),
        NoRedirectHandler(),
        _DeadlineHTTPHandler(deadline),
    ]
    if _DeadlineHTTPSHandler is not None:
        handlers.append(_DeadlineHTTPSHandler(deadline))
    return handlers


def open_direct(request, timeout):
    """Open ``request`` without ambient proxies or redirects within ``timeout``.

    ``timeout`` is the wall-clock budget for the whole request: connection,
    request upload, response headers, and every response body read.
    """
    opener = urllib.request.build_opener(*direct_handlers(timeout))
    return opener.open(request, timeout=timeout)


def parse_json_object(raw, label):
    """Decode strict UTF-8 JSON that rejects duplicate keys at every depth."""
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except RecursionError as exc:
        raise ValueError(f"{label} JSON response is too deeply nested") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON response must be an object")
    return value


def read_json_object(response, request, *, limit, deadline, label, require_target=True):
    """Validate one direct response and return its bounded JSON object."""
    if response.getcode() != 200:
        raise ValueError(f"{label} returned a non-success status")
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise ValueError(f"{label} response is not application/json")
    geturl = getattr(response, "geturl", None)
    if geturl is None:
        if require_target:
            raise ValueError(f"{label} response target is unavailable")
    elif geturl() != request.full_url:
        raise ValueError(f"{label} response target changed")
    raw = response.read(limit + 1)
    _remaining(deadline)
    if len(raw) > limit:
        raise ValueError(f"{label} response exceeded the safe byte limit")
    return parse_json_object(raw, label)
