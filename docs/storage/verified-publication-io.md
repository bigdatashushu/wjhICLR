# Incremental framework IO and verified publication reuse

`FrameworkStateIO` enables `StateWorkspace.incremental_leases` inside the existing
fixed workspace ceiling. Each exclusive section reads a small durable accounting
record, fsyncs a dirty intent, accounts actual changes to touched paths, then
persists clean logical and allocated totals. It does not update workspace LRU or
scan unchanged members. A crashed owner leaves a dirty record; the next owner
reconciles the tree before admitting writes. A raw exclusive `HotCache.lease`
invalidates the counters. Raw writers must acquire that lease; incremental mode
requires all ordinary mutations to use `WorkspaceIO`.
Accounting files have their own persistent allocation reservation. Their atomic
replacement holds the short global metadata lock. Failed file/directory fsync
keeps the dirty intent; recovery synchronizes the tree before trusting its totals.

The physical allocation and logical ceilings, per-write filesystem free-space
floor, old-file-plus-temporary peak, atomic rename and file/directory fsync remain
enforced. `StateWorkspace` caches the immutable baseline index while its file
identity and journal anchors remain unchanged.

`SceneHotInputProvider` enables `reuse_verified_publications()` on a publication
provider with an approved physical descriptor. Incoming scene members are checked
once during transport; the complete archive member set is checked before publish.
The immutable publication catalog and descriptor supply the original anchors.
Solving decodes RGB PNGs and the selected binding directly. It does not rehash
geometry/masks, rebuild migration proofs, or reload the source and derived bindings
for every question in the scene. The ordinary publication provider remains
available for independent auditing.

Derived replay loads only the requested approved reference. A framework-created
`ProducerVerifiedBindingContext` binds that decoded object to its frozen request.
The consumer still checks question, scene, parameters, seed, and optional model
identity; original generation and derivation receipts remain unchanged. This
capability is not a generated-code tool.

Offline episode records reuse the original PNG encoding from the decoded scene
cache. Encoding happens outside the workspace lock; the 32 frame files are written
in one admitted IO section. Cache entries stay scene-bounded and caller episode
objects remain isolated.

Verification uses synthetic publications, concurrent quota competition, process
death and reconciliation, and legacy audit paths. Performance measurements exclude
models and API requests. Deploy this version from its own checkout; active frozen
experiment source trees retain their recorded identity.
