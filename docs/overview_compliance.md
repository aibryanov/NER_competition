# Overview Compliance Audit

## 1. Initial audit

Before the refactor, the repository only partially matched the competition task.

### What did not match the task

1. The main training pipeline used `RuNNE`, not the competition data from `competition/train_dataset.tsv`.
2. Evaluation was done on token chunks, while the competition metric is strict match on character spans and labels.
3. There was no reliable end-to-end path from model output to competition submission format.
4. The submission example requires the column name `Prediction`, but the repository did not enforce this contract.
5. The repository still contained code paths and artifacts for external datasets that are irrelevant for the competition setup.

### What already matched

1. The core model class was a valid sequence-labeling architecture for NER: BiLSTM + optional CRF.
2. The code already used token-level labels and could be adapted to BIOES tagging.
3. The repository already had local competition train and test files.

### Conclusion

After the audit, the repository was assessed as:

\[
\text{partially compliant with Task 1, non-compliant with the official evaluation contract}
\]

It did not solve Task 2 either, because the production real-time masking / demasking architecture was not described.

## 2. Changes made

### Code and structure

1. Replaced external-dataset loading with competition-only loading from local files.
2. Added a dedicated parser for `train_dataset.tsv` and `private_test_dataset.csv`.
3. Refactored `main.py` into a CLI with two commands:
   - `train`
   - `submit`
4. Added structured modules for:
   - pipeline assembly,
   - training,
   - evaluation,
   - inference,
   - runtime and logging.
5. Removed repository artifacts that were not part of the competition solution:
   - exploratory notebooks,
   - the old embedding download helper,
   - code paths for unrelated datasets.

### Metric alignment

1. Dev/test evaluation was changed from token-chunk comparison to strict character-span comparison.
2. Predicted token chunks are now mapped back to character offsets using stored token boundaries.
3. The submission writer now produces:

```text
id,Prediction
```

where `Prediction` is the string representation of:

\[
[(start, end, category), \dots]
\]

or `[]`.

### Data alignment

1. The training pipeline now reads `target` from the competition train file and normalizes it to:

\[
\{(s_i, e_i, y_i)\}_{i=1}^n
\]

2. Rare nested spans with the same label are collapsed to the outer span before BIOES conversion, because a single-layer sequence tagger cannot represent two overlapping labels on the same token positions.

### Empirical note about the provided training data

The local competition train file was inspected directly. The result:

\[
\text{rows} = 8287
\]

\[
\text{overlap pairs} = 53
\]

\[
\text{rows with overlaps} = 44
\]

and all observed overlaps were nested spans with the same label.

This is exactly why outer-span collapse is a valid projection for this repository.

## 3. Why the metric is now correct

The competition evaluates strict match on span boundaries and labels. Let:

\[
G = \{(s_i, e_i, y_i)\}_{i=1}^{N_G}
\]

be the gold set of entities for all examples, and let:

\[
P = \{(\hat{s}_j, \hat{e}_j, \hat{y}_j)\}_{j=1}^{N_P}
\]

be the predicted set.

A predicted entity is correct if and only if:

\[
(\hat{s}_j, \hat{e}_j, \hat{y}_j) \in G
\]

Therefore:

\[
TP = |P \cap G|
\]

\[
FP = |P \setminus G|
\]

\[
FN = |G \setminus P|
\]

and the competition metric is:

\[
\mathrm{Precision} = \frac{TP}{TP + FP}
\]

\[
\mathrm{Recall} = \frac{TP}{TP + FN}
\]

\[
\mathrm{F1} = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
\]

The refactored evaluation computes exactly these quantities on character spans, so it matches the official task definition.

## 4. Why character-span reconstruction is mathematically valid

Let the tokenizer produce a sequence of tokens with character offsets:

\[
T = \{(w_k, a_k, b_k)\}_{k=1}^{m}
\]

where token \(w_k\) occupies the half-open interval:

\[
[a_k, b_k)
\]

If BIOES decoding predicts an entity spanning token indices from \(u\) to \(v\) inclusive, then the reconstructed character span is:

\[
[a_u, b_v)
\]

This is correct because the entity is represented by the contiguous union of token intervals:

\[
[a_u, b_u) \cup [a_{u+1}, b_{u+1}) \cup \dots \cup [a_v, b_v)
\]

Under a monotone tokenizer with preserved order, this union corresponds exactly to the outer character boundaries:

\[
[a_u, b_v)
\]

So the span reconstruction used in evaluation and submission is the correct projection from token-level decoding back to the character-level metric space.

## 5. Why collapsing rare nested same-label spans is valid

The training target for a standard sequence tagger assigns exactly one label to each token:

\[
z_k \in \mathcal{Y}
\]

If two spans overlap on the same token positions, then a single-label model would require:

\[
z_k = y_a \quad \text{and} \quad z_k = y_b
\]

simultaneously, which is impossible unless the targets are identical in a way that can be merged.

In the competition train file, the observed overlaps are rare and were empirically found to be same-label nested spans. In that case, replacing

\[
(s_1, e_1, y), (s_2, e_2, y)
\]

with the outer span

\[
(\min(s_1, s_2), \max(e_1, e_2), y)
\]

preserves the category and maximizes boundary coverage inside a single-label BIOES representation.

This makes the projection consistent with the hypothesis class of the model.

## 6. Vision for Task 2

The repository is code for Task 1, but the overview also requests a production concept. A compact version of that concept is:

1. The client request first goes through a low-latency NER service.
2. The service returns detected spans:

\[
\{(s_i, e_i, y_i)\}_{i=1}^{n}
\]

3. These spans are replaced with deterministic placeholders such as:

\[
\texttt{<PII\_PHONE\_1>}, \texttt{<PII\_PASSPORT\_1>}
\]

4. The masked text is sent to the LLM.
5. A secure in-memory vault stores the mapping:

\[
\texttt{placeholder} \mapsto \texttt{original value}
\]

6. While the LLM response is streamed, a post-processor replaces placeholders back with the original values before the text is shown to the user.

This architecture satisfies the business requirements because:

1. PII does not leave the protected boundary in raw form.
2. The NER pass is a single fast model invocation, which is compatible with real-time processing.
3. New categories can be added by extending the label set and retraining the model.

Formally, let the original text be \(x\), let \(S\) be the detected entity set, and let \(M\) be the masking operator. If each detected entity is mapped to a unique placeholder \(p_i\), then:

\[
M(x, S) = x'
\]

and the secure vault stores:

\[
V = \{p_i \mapsto v_i\}_{i=1}^{|S|}
\]

If placeholders are unique and non-overlapping, then demasking \(D\) satisfies:

\[
D(M(x, S), V) = x
\]

This is the formal reason the masking / demasking stage is information-preserving for authorized post-processing, while still preventing raw PII from being sent to the LLM.

## 7. What remains outside the repository scope

The repository now matches Task 1 significantly better, but Task 2 from the overview is still conceptual rather than implemented in code.

In particular, the following production concerns are not fully implemented:

1. streaming-safe masking / demasking in a live LLM interaction loop;
2. secure ephemeral storage of original sensitive values;
3. a real-time low-latency serving architecture with monitoring and rollback.

These should be described as a separate system design document or presentation.
