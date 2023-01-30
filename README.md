# Security-Preserving Federated Learning via Byzantine-Sensitive Triplet Distance

This is an official implementation of the following paper:
> Youngjoon Lee, Sangwoo Park, and Joonhyuk Kang.
**[Security-Preserving Federated Learning via Byzantine-Sensitive Triplet Distance](https://arxiv.org/abs/2210.16519)**  
_arXiv preprint arXiv:2210.16519_.

## Requirements
The implementation runs on

```bash docker.sh```

Additionally, please install the required packages as below

```pip install tensorboard medmnist```

## Byzantine attacks
This paper considers the following poisoning attacks
- Targeted model poisoning ([Bhagoji, Arjun Nitin, et al. ICML 2019](https://arxiv.org/abs/1811.12470)): Targeted model poisoning attack for federated learning
- MPAF ([Xiaoyu Cao, Neil Zhenqiang Gong. CVPR Workshop 2022](https://arxiv.org/abs/2203.08669)): Untargeted model poisoning attack for federated learning

## Byzantine-Robust Aggregation Techniques
This paper considers the following Byzantine-Robust aggregation techniques
- Vanilla ([McMahan, Brendan, et al. AISTATS 2017](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com))
- Krum ([Blanchard, Peva, et al. NIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html))
- Trimmed-mean ([Yin, Dong, et al. ICML 2018](https://proceedings.mlr.press/v80/yin18a))
- Fang ([Fang, Minghong, et al. USENIX 2020](https://arxiv.org/abs/1911.11815))

## Dataset
- Blood cell classification dataset ([Andrea Acevedo, Anna Merino, et al. Data in Brief 2020](https://www.sciencedirect.com/science/article/pii/S2352340920303681))

## Experiments
Without Byzantine attacks experiment runs on

```bash execute/run0.sh```

Impact of Byzantine percentage runs on

```bash execute/run1.sh```

Impact of non-iid degree runs on

```bash execute/run2.sh```

## Acknowledgements
Referred http://doi.org/10.5281/zenodo.4321561
