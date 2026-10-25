# Historical r04 SSD publication transport

`scripts/cluster_ssd_publication_transport.py` preserves the transport adapter
developed for the 230-scene r04 task. Its production entry point requires the
original method hash, exact frozen dependencies, the explicit SSD directory,
and an activation receipt from the task's operator. It does not stop the
previous coordinator or restart model workers.

`scripts/cluster_publication_backpressure_v2.py` is the byte-identical v2
dependency retained with the adapter. Its SHA-256 is
`0f8bcfc192fbac00faf69a91c564a43bb8e9281bc1a69be851fb5c962f79f787`.
The v3 loader checks this hash before executing the dependency.

The local tests resolve this dependency inside the repository instead of
depending on a particular user's output directory. Their source pins identify
the current repository coordinator and worker modules; production validation
continues to use the original task's `source_frozen` pins and method identity.

Run the fixture tests from a development environment:

```bash
python -m pytest tests/unit/test_cluster_ssd_publication_transport.py
```

These tests use temporary SQLite queues and loopback HTTP servers. They do not
need GPUs, model requests, or access to a running task.

Including these files in `main` does not deploy them. They are historical r04
tools and do not support the current 288-scene full-question task's method
identity. Any deployment remains a separate operator action.
