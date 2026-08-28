# Extending AgriTune

_Placeholder — expands as the core protocols (Phase 1) stabilize._

The v1 layer boundary (`EncoderBackend → EncoderGateway → FeatureStore/FeatureProvider → Task →
Trainer`, see [architecture.md](architecture.md)) is designed so a new downstream task
(classification, regression, depth, detection — `agritune_implementation_plan.md` §26 `v0.4`) only
needs a new `Task`/`Decoder` implementation against the existing `EncoderFeatures` contract, and a
new tracking backend only needs to implement the `Tracker` protocol
(`precisionai.agritune.tracking`). Neither requires touching the trainer or the encoder/gateway
layer.
