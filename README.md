# ADAPT MLIP

<p>
  <img src="banditLogo.png" alt="Bandit logo" width="33%" align="left">
  This project implements a machine learning interatomic potential (MLIP) / machine learning force field (MLFF) for atomistic structures. It consumes Cartesian-coordinate representations directly instead of graph-based inputs, using a configurable Transformer encoder backbone inspired by "ADAPT: Lightweight, Long-Range, Machine Learning Force Fields Without Graphs". The same architecture can be trained for direct energy/force prediction or for energy-conserving force prediction, where forces are obtained from the gradient of predicted energy with respect to coordinates.
</p>

<br clear="left">

## Quick Start

Every run must specify a formulation and at least one requested physical quantity. Output heads may be provided explicitly, but if omitted the CLI selects defaults: `atomwise_energy` for requested energy and `direct_force` for requested force in direct models. Conservative models request both `energy` and `force`, but use only an energy head because force is computed by autograd.

Direct energy-and-force training on tiny OMOL data:

```bash
python adapt_cli.py \
  --formulation direct \
  --requested-quantities energy force \
  --train-data-path data/tiny_omol_data/train_sample.xyz \
  --validation-data-path data/tiny_omol_data/test_sample.xyz \
  --test-data-path none \
  --epochs 3
```

Direct energy-and-force training with explicit heads:

```bash
python adapt_cli.py \
  --formulation direct \
  --requested-quantities energy force \
  --heads atomwise_energy direct_force \
  --train-data-path data/tiny_omol_data/train_sample.xyz \
  --validation-data-path data/tiny_omol_data/test_sample.xyz \
  --test-data-path none \
  --epochs 3
```

Energy-conserving training:

```bash
python adapt_cli.py \
  --formulation conservative \
  --requested-quantities energy force \
  --heads atomwise_energy \
  --train-data-path data/tiny_omol_data/train_sample.xyz \
  --validation-data-path data/tiny_omol_data/test_sample.xyz \
  --test-data-path none \
  --epochs 3
```

If `--validation-data-path none` is passed, the training loop reserves the last 500 structures from the training dataset for validation. This fallback requires at least 501 training structures. If `--validation-data-path` is omitted entirely, the value from `TrainingConfig.validation_data_path` is used.

## Resuming Training

Training uses a "train until N total epochs" convention. If a saved run already contains 45 completed epochs and the next command sets `--epochs 60`, the code trains 15 additional epochs, from epoch 46 through epoch 60. The saved checkpoint, optimizer state, objective loss-scale state, and history are loaded when `--saved-model-path` points to either a checkpoint file or a saved-run directory.

Example first run:

```bash
python adapt_cli.py \
  --formulation direct \
  --requested-quantities force \
  --train-data-path data/tiny_omol_data/train_sample.xyz \
  --validation-data-path data/tiny_omol_data/test_sample.xyz \
  --test-data-path none \
  --results-output-path ./outputs/tiny_force_epochs45/ \
  --val-eval-freq 5 \
  --rot-augmentation \
  --epochs 45
```

Resume that run to 60 total epochs and write the new checkpoint/history to a different output directory:

```bash
python adapt_cli.py \
  --formulation direct \
  --requested-quantities force \
  --train-data-path data/tiny_omol_data/train_sample.xyz \
  --validation-data-path data/tiny_omol_data/test_sample.xyz \
  --test-data-path none \
  --saved-model-path ./outputs/tiny_force_epochs45/ \
  --results-output-path ./outputs/tiny_force_epochs60/ \
  --val-eval-freq 5 \
  --rot-augmentation \
  --epochs 60
```

With `--val-eval-freq 5`, the resumed example evaluates at epochs 50, 55, and 60. It does not evaluate at epoch 46 because evaluation is triggered by `epoch % val_eval_freq == 0`.

## Evaluation and Force Agreement

Use `evaluation.py` to load a saved model and evaluate losses plus metric-registry values on a labeled `.xyz` or `.adapt.pt` dataset. The script uses the same objective API as training, so direct and conservative models use their own evaluation-time forward behavior.

Evaluate a direct energy-and-force model:

```bash
python evaluation.py \
  --model-path ./outputs/adapt_runtime.pth \
  --data-path data/tiny_omol_data/test_sample.xyz \
  --formulation direct \
  --requested-quantities energy force \
  --heads atomwise_energy direct_force \
  --batch-size 32 \
  --device auto
```

Evaluate an energy-conserving model:

```bash
python evaluation.py \
  --model-path ./outputs/adapt_runtime.pth \
  --data-path data/tiny_omol_data/test_sample.xyz \
  --formulation conservative \
  --requested-quantities energy force \
  --heads atomwise_energy \
  --batch-size 32 \
  --device auto
```

When `--model-path` is a saved-run directory instead of a `.pth` file, pass the directory and, if needed, the matching runtime name:

```bash
python evaluation.py \
  --model-path ./outputs/tiny_force_epochs60/ \
  --runtime-name adapt_runtime \
  --data-path data/tiny_omol_data/test_sample.xyz \
  --formulation direct \
  --requested-quantities force \
  --heads direct_force
```

Use `force_agreement.py` to check whether predicted forces are consistent under random rotations. The script predicts forces on the original structures, predicts again after randomly rotating the coordinates, rotates the second set of predicted forces back to the original frame, and reports the disagreement between the two predictions.

```bash
python -m results_code.force_agreement \
  --model-path ./outputs/adapt_runtime.pth \
  --data-path data/tiny_omol_data/test_sample.xyz \
  --formulation direct \
  --requested-quantities energy force \
  --heads atomwise_energy direct_force \
  --batch-size 32 \
  --device auto \
  --random-seed 42
```

The main reported value is `mae_per_component`, which is the mean absolute difference between the original force prediction and the unrotated augmented-force prediction over valid atoms and Cartesian components. `mean_vector_norm` reports the average per-atom vector disagreement.

## Output Type and Formulation Constraints

The code separates physical quantities from concrete output heads. Physical quantities are `energy` and `force`; output heads are implementation choices that produce one of those quantities.

Valid combinations:

| Formulation | Requested quantities | Valid heads |
| --- | --- | --- |
| `direct` | `energy` | One energy head: `atomwise_energy` or `scalar_decoder_energy` |
| `direct` | `force` | `direct_force` |
| `direct` | `energy force` | One energy head plus `direct_force` |
| `conservative` | `energy force` | Exactly one energy head; no force head |

Invalid combinations include:

- A direct model whose heads do not produce exactly its requested quantities.
- A conservative model requesting only energy or only force.
- A conservative model containing `direct_force`.
- Selecting two heads that produce the same quantity, such as both energy heads.
- Repeating a requested quantity or head type.

## CLI Reference

The CLI is intended as an easy-start interface. Some values are intentionally config-file-only: `learning_rate`, `deterministic`, `random_seed`, `norm_stats_path`, all `EmbeddingConfig` values, and all `EncoderConfig` values except `attention` and `activation`.

All available CLI arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--formulation FORMULATION` | Yes | Objective formulation, currently `direct` or `conservative`. |
| `--requested-quantities {energy,force} ...` | Yes | Physical quantities included in the objective. |
| `--heads {atomwise_energy,direct_force,scalar_decoder_energy} ...` | No | Concrete output heads. Defaults are inferred from requested quantities and formulation if omitted. |
| `--train-data-path PATH` | No | Training `.xyz` or `.adapt.pt` file. XYZ files are converted if needed. |
| `--validation-data-path PATH_OR_NONE` | No | Validation `.xyz` or `.adapt.pt` file. Use `none` to reserve validation data from the training dataset. |
| `--test-data-path PATH_OR_NONE` | No | Optional test `.xyz` or `.adapt.pt` file. Use `none` for no test set. |
| `--saved-model-path PATH_OR_NONE` | No | Optional checkpoint file or saved-run directory to resume from. |
| `--results-output-path PATH` | No | Directory for checkpoint and history outputs. |
| `--runtime-name NAME` | No | Base name for checkpoint and history files. |
| `--epochs EPOCHS` | No | Total epoch count to train until, including loaded history when resuming. |
| `--batch-size BATCH_SIZE` | No | Number of structures per training/evaluation batch. |
| `--device DEVICE` | No | Device string such as `auto`, `cpu`, `cuda`, or `cuda:0`. |
| `--val-eval-freq N` | No | Evaluate validation, and test if present, every `N` epochs. |
| `--rot-augmentation` | No | Enable random rotation augmentation during training. |
| `--no-rot-augmentation` | No | Disable random rotation augmentation during training. |
| `--attention ATTENTION` | No | Encoder and head-encoder attention implementation. |
| `--activation ACTIVATION` | No | Activation function used by configured MLP/FFN layers. |
| `-h`, `--help` | No | Print CLI help. |

## Architecture Options

Available attention mechanisms:

| Name |
| --- |
| `standard_mha` |
| `gqa_attention` |
| `nlv_attention` |
| `mlv_attention` |
| `mlv_allresid_attention` |
| `allresid_attention` |
| `qek_attention` |

Available activation functions:

| Name |
| --- |
| `relu` |
| `leaky_relu` |
| `gelu` |
| `silu` |
| `tanh` |
| `sigmoid` |

Available output heads:

| Head | Produces |
| --- | --- |
| `atomwise_energy` | `energy` |
| `scalar_decoder_energy` | `energy` |
| `direct_force` | `force` |

Available embedding types:

| Config field | Options | Notes |
| --- | --- | --- |
| `coordinate_embedding` | `coord_mlp` | Coordinate MLP for per-atom Cartesian coordinates. |
| `species_embedding` | `Z_mlp`, `Z_table` | Species MLP or atomic-number lookup table. |
| `charge_embedding` | `charge_mlp`, `charge_table` | Structure-level charge token. |
| `spin_multiplicity_embedding` | `spin_multiplicity_table` | Structure-level spin multiplicity token, valid values `0..11`. |

Embedding choices are currently configured in `configs/arch_config.py`, not from the CLI.

## Data Formatting and Processing

The library accepts `.xyz` and `.adapt.pt` dataset paths. When given an `.xyz` file, the resolver checks for a corresponding `.adapt.pt` file in the same directory and creates one if needed. The `.adapt.pt` layout stores flattened atom tensors plus structure offsets, which makes loading and batching independent of molecule size.

Input files must use a consistent extended-XYZ layout. Every structure must contain exactly one species field and one three-component coordinate field, and all structures must use the same property names, property widths, and header fields. Optional targets must be present consistently: either every structure contains forces or none do, and either every structure contains energy or none do. Forces must be three-component per-atom values, while energy must be a single structure-level header value.

`charge` and `spin_multiplicity` are structure-level header values. Each must either be present for every structure or absent from every structure. When present, both must be written as integers; floating-point spellings such as `0.0` are not accepted. If either field is omitted from the entire file, the converter emits a warning and assigns integer `0` to every structure. Element species must be recognized chemical symbols, and each atom row must contain exactly the number of values declared by the `Properties` field.

Use the malformed-XYZ checker before conversion:

```bash
python data/check_xyz_malformed.py path/to/data.xyz
```

## Code Design

- `training_coordination.py` handles top-level training control flow: dataset resolution/loading, normalization, model construction, objective construction, training, validation, testing, checkpointing, and history writing.
- `MLIP_arch/` defines the model architecture, including embeddings, attention implementations, encoder backbone, output heads, and the model factory.
- `objectives/` defines formulation-specific forward/loss behavior for direct and energy-conserving training.
- `data/` handles XYZ conversion, `.adapt.pt` datasets, normalization, augmentation utilities, and data validation.


## Citation

If you use our work, please consider citing:

```bibtex
@article{dramko2025adapt,
  title   = {ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs},
  author  = {Dramko, Evan and Xiong, Yihuang and Zhu, Yizhi and Hautier, Geoffroy and Reps, Thomas and Jermaine, Christopher and Kyrillidis, Anastasios},
  journal = {arXiv preprint arXiv:2509.24115},
  year    = {2025}
}
```

## Licensing
This code is released subject to an MIT License. Further details are in the attached licensing file. 
