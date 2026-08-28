# Feature caching

_Placeholder — expands as Phase 6/7 (feature store, feature providers) land._

Feature cache keys fingerprint sample ID, image hash, augmentation configuration/seed, encoder
model, encoder revision, encoder preprocessing, and feature schema version — see
[encoder.md](encoder.md) for why the encoder revision component is best-effort rather than
authoritative. `DirectoryFeatureStore` is the development-scale store; `ShardedFeatureStore` is the
production store. Precomputation (`agritune features build`) is resumable — restarting after a
partial run only encodes the missing samples. See `agritune_implementation_plan.md` §9–10.
