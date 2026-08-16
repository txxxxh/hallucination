# Scientist profiles multimodal features

The gzip artifact is split into GitHub-compatible chunks. Reassemble and verify it with:

```bash
cat base_multimodal_features.jsonl.gz.part-* > base_multimodal_features.jsonl.gz
cat intervention_multimodal_features.jsonl.gz.part-* > intervention_multimodal_features.jsonl.gz
sha256sum -c SHA256SUMS
gzip -t base_multimodal_features.jsonl.gz
gzip -t intervention_multimodal_features.jsonl.gz
```

Artifact sizes:

- `base_multimodal_features.jsonl.gz`: 312,203,553 bytes (1,069,514,867 bytes uncompressed)
- `intervention_multimodal_features.jsonl.gz`: 2,697,815,105 bytes
