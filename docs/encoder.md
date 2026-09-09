# Encoder

AgriTune never trains the encoder — it only consumes features from it. All encoder access goes
through the `EncoderBackend` protocol (`precisionai.agritune.encoder.base`), which has two
implementations:

- `FakeEncoderBackend` — configurable patch dimension, CLS dimension, patch grid, latency, and
  injectable failures (429, 500, timeout, malformed response). This is what unit tests and CI run
  against; no live API key is ever required for the default test suite.
- `RemoteEncoderBackend` — wraps the real hosted embedding API.

## The real API contract

The hosted encoder is an OpenAI-SDK-compatible embeddings service. AgriTune calls it exactly as:

```python
from openai import OpenAI

client = OpenAI(base_url="<deployment>/v1", api_key="sk-pai-...")

resp = client.embeddings.create(
    model="pai-embedding",              # public alias; server resolves the real backing model
    input=data_uri,                     # "data:image/...;base64,..." or http(s) URL
    encoding_format="base64",            # little-endian float32 buffer, more efficient than nested lists
    extra_body={
        "return_patch_tokens": True,     # Precision AI extension, not part of the OpenAI API
        "native_resolution": True,       # optional — grid becomes per-image and often non-square
    },
)
```

Response shape, per item in `resp.data`:

- **`embedding`** — the CLS/pooled vector, `(D_cls,)`, float32. `D_cls` is model-dependent and
  learned at runtime from the response, never hardcoded.
- **`patch_embeddings`** — patch tokens, **channels-first** `(D_patch, H, W)`, base64 little-endian
  float32 (when `encoding_format="base64"`). Decode with
  `np.frombuffer(base64.b64decode(x), dtype="<f4").reshape(patch_shape)`, then transpose/flatten to
  the `[N, D_patch]` layout AgriTune's `EncoderFeatures.patch_tokens` expects (`N = H * W`).
- **`patch_shape`** — `[D_patch, H, W]`. `H`/`W` are frequently **not equal** when
  `native_resolution` is enabled — never assume a square grid.

### What the API does *not* expose

There is no request- or response-level **encoder revision** field, and no dimensions in the
`GET /v1/models` listing available to non-privileged callers — the server can change the model
backing the `pai-embedding` alias without any client-visible version bump. `RemoteEncoderBackend`
therefore fingerprints the encoder from **the configured model alias plus the dimensions actually
observed on the first response of a run** (`D_cls`, `D_patch`, dtype). Document this explicitly to
users: if the server silently swaps the model behind an alias without changing observed
dimensions, AgriTune has no way to detect it, and cached features would silently mix encoder
versions. Pin to a concrete internal model id (privileged credentials) if this matters for a given
deployment.

### Auth, errors, rate limits

- Auth is a bearer token via the SDK's `api_key` (`sk-pai-...` or a Cognito access token) — no
  custom header.
- Errors follow an OpenAI-style envelope (`{"error": {"message", "type", "param", "code"}}`) with
  standard status codes; `429` responses carry a `Retry-After` header that the gateway's retry
  policy must honor over its own backoff schedule when present.
- The server enforces **quota** (daily/monthly item and request counts), not a documented
  requests-per-minute limit — `agritune encoder benchmark` measures achievable throughput
  empirically rather than relying on a published rate.
- The server does not enforce a request timeout; `RemoteEncoderBackend` must always set an
  explicit client-side timeout (`encoder.timeout.request_seconds`).

See [feature-caching.md](feature-caching.md) for how the encoder fingerprint feeds the
feature cache key.
