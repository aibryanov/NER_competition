# AlfaBank NER Competition

This repository is aligned to the AlfaBank NER competition setup:

- training data is read from `competition/train_dataset.tsv`;
- inference data is read from `competition/private_test_dataset.csv`;
- the model predicts character spans and writes submission files with columns `id,Prediction`.

## What is implemented

- local competition-only data loading;
- BIOES sequence tagging with a BiLSTM-CRF;
- strict evaluation on character spans and categories;
- submission generation in the expected format `[(start, end, 'Category')]`.

## What you need before running

Place a reduced fastText model into:

```text
word_embeddings/fasttext_ru_50dim.bin
```

The current default config expects `50` dimensions and Russian embeddings.

## Train

```bash
.venv/bin/python main.py train
```

The best checkpoint is saved to:

```text
models/competition_bilstm_crf
```

This checkpoint now contains everything needed for loading and inference:

- model weights;
- token/char/tag mappings;
- pretrained word embeddings matrix;
- saved `model` and `preprocessing` settings.
- saved `regex.enabled_labels` settings for hybrid inference.

Validation runs at the end of every epoch, and the checkpoint is updated only when `DEV F1` improves.

Sentence-level entity presence head is configured in `cfg.sentence_entities` inside `src/config.py`. It is trained jointly with the token-level NER objective and predicts which entity classes are present anywhere in the sentence.

## Generate submission

```bash
.venv/bin/python main.py submit
```

The submission file is saved to:

```text
models/competition_bilstm_crf_submission.csv
```

`submit` can run directly from the full checkpoint `models/competition_bilstm_crf`.
Legacy checkpoints are still supported through `mappings/mapping.pkl`.

## Evaluate per class

```bash
.venv/bin/python main.py evaluate --split dev
```

You can also point to another labeled TSV file:

```bash
.venv/bin/python main.py evaluate --dataset-path path/to/labeled.tsv --split all
```

To evaluate a specific checkpoint:

```bash
.venv/bin/python main.py evaluate --checkpoint-path models/model_96056 --split dev
```

For legacy checkpoints without embedded mappings, you can also pass:

```bash
.venv/bin/python main.py evaluate --checkpoint-path models/model_96056 --mapping-path mappings/mapping.pkl --split dev
```

The report prints per-class `Precision / Recall / F1` together with `TP / Pred / Gold` in a text table.

## Hybrid model + regex

Regex rules are applied only to the labels listed in `src/config.py` under `cfg.regex.enabled_labels`. Those labels are taken over by regex completely; all remaining classes are predicted by the model.

Current default regex labels are:

- `Номер телефона`
- `Сведения об ИНН`
- `Паспортные данные`
- `Номер банковского счета`
- `Номер карты`
- `Одноразовые коды`
- `Email`

To evaluate the hybrid method from a checkpoint:

```bash
.venv/bin/python main.py evaluate --checkpoint-path models/competition_bilstm_crf --split dev
```

To evaluate a legacy `state_dict` together with a separate mapping:

```bash
.venv/bin/python main.py evaluate --checkpoint-path models/model_96056 --mapping-path mappings/mapping.pkl --split dev
```

## Load model from files

If you have a saved model file and a separate mapping file, you can load them directly:

```python
from src.inference.competition import load_model_and_mapping_from_files

model, mapping = load_model_and_mapping_from_files(
    checkpoint_path="models/model_96056",
    mapping_path="mappings/mapping.pkl",
)
```

The same pair can be used for submission generation:

```bash
.venv/bin/python main.py submit --checkpoint-path models/model_96056 --mapping-path mappings/mapping.pkl
```

## Audit and math notes

See:

- `docs/overview_compliance.md`
