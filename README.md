# ADAPT Machine Learning Interatomic Potential
This repository contains the necessary code for training or conducting inference with the ADAPT machine learning model. The associated [paper](https://arxiv.org/abs/2509.24115), [website](https://evandramko.github.io/ADAPT_webpage/), and [dataset](https://zenodo.org/records/17411327) for the ADAPT architecture can provide further resources. 

ADAPT supports CLI runtimes for training and evaluation through `adapt_cli.py`, and also exposes Python APIs for custom workflows. 

## Release Notes

### v0.3.0 – Major Refactor

This release introduces a significant redesign of the codebase. Several behaviors have changed compared to previous versions.

#### Changed
- The model **no longer predicts energies**; it now predicts **forces only**.
- Structural input is now provided via **`.xyz` or `.extxyz` files** instead of serialized `.pt` files.

#### Added
- **Command-line interface (CLI)** support for training and inference workflows.

#### Removed
- Energy prediction support (temporarily).

#### Planned
- Multi-GPU training support (currently limited to single-GPU).
- Reintroduction of energy prediction in a future release.

## Command Line Usage

You can run training and evaluation directly from the command line without editing config files:

```bash
# Train
python adapt_cli.py train --train-path data/training_data.xyz --test-path data/training_data.xyz --is-crystal

# Train and force fresh normalization statistics (overwrite existing norm_stats)
python adapt_cli.py train --train-path data/training_data.xyz --test-path data/test_100_set.xyz --is-crystal --recompute-stats

# Evaluate one frame
python adapt_cli.py eval --path data/training_data.xyz --frame-idx 0 --is-crystal

# Evaluate all frames
python adapt_cli.py eval --path data/training_data.xyz --all-frames --is-crystal
```

## Tutorials

1. [Installation]()--Pip installable version coming soon!
2. [ADAPT Architecture](tutorials/architecture.md)
3. [Training a model](tutorials/training.md)
4. [Deplyment and Inference Time](tutorials/inference.md)


## Citation

If you use this code or the ADAPT model in your research, please cite:

```bibtex
@article{dramko2025adapt,
  title   = {ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs},
  author  = {Dramko, Evan and Xiong, Yihuang and Zhu, Yizhi and Hautier, Geoffroy and Reps, Thomas and Jermaine, Christopher and Kyrillidis, Anastasios},
  journal = {arXiv preprint arXiv:2509.24115},
  year    = {2025}
}
