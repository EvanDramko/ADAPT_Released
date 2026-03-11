# ADAPT Machine Learning Interatomic Potential
This repository contains the necessary code for training or conducting inference with the ADAPT machine learning model. The associated [paper](https://arxiv.org/abs/2509.24115), [website](https://evandramko.github.io/ADAPT_webpage/), and [dataset](https://zenodo.org/records/17411327) for the ADAPT architecture can provide further resources. 

ADAPT supports CLI runtimes for training and evaluation through `adapt_cli.py`, and also exposes Python APIs for custom workflows. 

The data samples included in the tiny_sample folder come from the OMol25 dataset released with Levine, Daniel S., et al. "The open molecules 2025 (omol25) dataset, evaluations, and models." arXiv preprint arXiv:2505.08762 (2025).

## Release Notes

### v0.3.1 – Major Refactor

This release introduces a significant redesign of the codebase. Several behaviors have changed compared to previous versions.

#### Changed
- The model **no longer predicts energies**; it now predicts **forces only**.
- Structural input is now provided via **`.xyz` or `.extxyz` files** instead of serialized `.pt` files.

#### Added
- **Command-line interface (CLI)** support for training and inference workflows.

#### Removed
- Energy prediction support (temporarily).

#### Planned
- Introduction of baseline default models. 
- Reintroduction of energy prediction in a future release.
- Multi-GPU training support (currently limited to single-GPU).


## Command Line Usage

You can run training and evaluation directly from the command line without editing config files. Any CLI argument you omit falls back to the corresponding value in the config file.

```bash
# Train
python adapt_cli.py train --train-path data/training_data.xyz --test-path data/training_data.xyz --is-crystal

# Train using a specific pretrained baseline
python adapt_cli.py train --train-path data/training_data.xyz --baseline-model saved_models/test_model.pth

# Evaluate one frame
python adapt_cli.py eval --path data/training_data.xyz --frame-idx 0

# Evaluate all frames (default when --frame-idx is omitted)
python adapt_cli.py eval --path data/training_data.xyz

# Evaluate with a specific checkpoint
python adapt_cli.py eval --path data/training_data.xyz --model-path saved_models/test_model.pth
```

### CLI Notes

- `train --train-path`: if omitted, uses `DataConfig.train_path`.
- `train --test-path`: if omitted, uses `DataConfig.test_path`; if that is `None`, training runs without evaluation.
- `train --baseline-model`: overrides `ModelPaths.pretrainPath` for loading a starting checkpoint.
- `train --device`: if omitted, uses `TrainConfig.device`.
- `train --augmentation/--no-augmentation`: overrides `TrainConfig.augmentation`.
- `train --is-crystal/--no-is-crystal`: overrides `DataConfig.isCrystal`.
- `eval --path`: if omitted, uses `DataConfig.test_path`.
- `eval --frame-idx`: if omitted, evaluation runs on all frames in the file.
- `eval --model-path`: overrides `ModelPaths.pretrainPath` for evaluation.
- `eval --device`: if omitted, uses `TrainConfig.device`.
- `eval --is-crystal/--no-is-crystal`: overrides `DataConfig.isCrystal`.

## Tutorials

1. [Installation]()--Pip installable version coming soon!
2. [ADAPT Architecture](https://evandramko.github.io/files/transformer.pdf)


## Citation

If you use this code or the ADAPT model in your research, please cite:

```bibtex
@article{dramko2025adapt,
  title   = {ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs},
  author  = {Dramko, Evan and Xiong, Yihuang and Zhu, Yizhi and Hautier, Geoffroy and Reps, Thomas and Jermaine, Christopher and Kyrillidis, Anastasios},
  journal = {arXiv preprint arXiv:2509.24115},
  year    = {2025}
}
