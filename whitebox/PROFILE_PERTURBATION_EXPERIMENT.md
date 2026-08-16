# Forward-only profile perturbation experiment

This experiment studies unsupervised internal-representation responses to
controlled prompt perturbations on `shuffled_prepend_profiles_question.json`.
It never calls `model.generate()`. Answer preference is measured with batched,
teacher-forced likelihoods of the two candidate names.

## Conditions

- `question_only`: no profiles; the base-failure condition.
- `full_context`: both complete profiles.
- `without_question_evidence`: removes exact question-mentioned attribute
  values from both profiles, avoiding answer-by-elimination from deleting a
  whole person.
- `minimal_decisive_evidence`: names plus only question-mentioned values.
- `profile_order_swap`: swaps the two complete profile blocks.
- `attribute_order_shuffle`: deterministically shuffles fields within profiles.
- `structure_only_context`: preserves names/schema/cardinality while masking
  values; a structural/length-oriented control (not token-exact matching).
- `negation_paraphrase`: meaning-preserving rule-based rewrite.
- `negation_flip`: semantic counterfactual. Phrase-level grammatical rewrites
  run first; unmatched forms use a deliberately broad first-`never`/`not`
  deletion (or `nor` to `and`). The original answer label is not treated as
  valid for this condition, and rewritten questions are retained for audit.
- `entity_paraphrase`: rule-based key-entity rewrite, with every rewrite logged.
- `structured_comparison_cue`: asks for constraint-by-constraint comparison.

Evidence matching and paraphrases are intentionally conservative and must be
audited in `prepared_conditions.json` before a full run.

## Run

Activate the existing environment:

```bash
source /home/tong56/whitebox/activate_whitebox.sh
```

Audit prompt construction without loading a model:

```bash
python /home/tong56/whitebox/profile_perturbation_unsupervised.py prepare \
  --limit 20 \
  --output /home/tong56/whitebox/profile_perturbation_forward_output
```

Extract a small smoke shard. The model and tokenizer are downloaded/cached only
under `/tmp/hf_profile_perturbation_cache`:

```bash
python /home/tong56/whitebox/profile_perturbation_unsupervised.py extract \
  --limit 10 \
  --output /home/tong56/whitebox/profile_perturbation_forward_output \
  --cache-dir /tmp/hf_profile_perturbation_cache
```

Extraction writes one compressed NPZ per item and resumes by default. Shards
may share an output directory if their offsets do not overlap.

Analyze the question-only failures for which full context restores the correct
candidate preference:

```bash
python /home/tong56/whitebox/profile_perturbation_unsupervised.py analyze \
  --output /home/tong56/whitebox/profile_perturbation_forward_output \
  --selection base_wrong_full_correct
```

Use `--selection base_wrong` for all question-only candidate-likelihood errors,
or `--selection all` to avoid behavioral cohort selection.

## Outputs

- `items/*.npz`: resumable per-item hidden and scalar response data.
- `run_config.json`: model, layer, data, and condition provenance.
- `analysis_summary.json`: PCA/GMM selection, bootstrap stability, and cluster
  response signatures.
- `cluster_assignments.csv`: item cluster and post-hoc correctness fields.
- `analysis_arrays.npz`: standardized features and PCA coordinates.

The clustering fit never receives `rgt_ans` or `wrg_ans`. Those labels are used
only for cohort selection and post-hoc summaries. Clusters should be described
as perturbation-response types, not established causal failure mechanisms.
