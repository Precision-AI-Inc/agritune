# Extending AgriTune

The v1 layer boundary (`EncoderBackend → EncoderGateway → FeatureStore/FeatureProvider → Task →
Trainer`, see [architecture.md](architecture.md)) is designed so a new downstream task
(classification, regression, depth, detection) only
needs a new `Task`/`Decoder` implementation against the existing `EncoderFeatures` contract, and a
new tracking backend only needs to implement the `Tracker` protocol
(`precisionai.agritune.tracking`) — see `JSONLTracker`, `TensorBoardTracker`, `MLflowTracker`,
`WandBTracker`, `NeptuneTracker`, and `CometTracker` for the pattern: an optional dependency raises
a clear `ImportError` with an install hint if missing, never a silent fallback. Neither kind of
extension requires touching the trainer or the encoder/gateway layer.

A new `FeatureProvider` (see [feature-caching.md](feature-caching.md) for the existing
cached/online/hybrid/prefetching set) only needs to satisfy `get_features(samples) ->
EncoderFeatures` — the trainer never knows or cares which one it was handed.

New routes on the API layer (`precisionai.agritune.api`) should call into
`precisionai.agritune.services` exactly like the CLI does — see
`precisionai.agritune.services.encoder_selection` for how encoder-backend selection is already
shared between the two entry points rather than duplicated.
