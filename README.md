# ADAPT Machine Learning Interatomic Potential
This repository contains the necessary code for training or conducting inference with the ADAPT machine learning model. The associated [paper](https://arxiv.org/abs/2509.24115), [website](https://evandramko.github.io/ADAPT_webpage/), and [dataset](https://zenodo.org/records/17411327) for the ADAPT architecture can provide further resources. 

ADAPT currently does not support CLI runtimes. The authors provide a template for a short python script that will efficiently load, check, and run inference time commands. Training is accomplished by running the appropriate python file. 

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
