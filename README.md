# Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm

This is an implementation for paper of [Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm"](https://openreview.net/forum?id=2dtU9ZbgSN&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions))
Jie Hao, Kaiyi Ji, Mingrui Liu, 37th Conference on Neural Information Processing Systems (NeurIPS 2023).

### Abstract
Coreset is a small set that provides a data summary for a large dataset, such that training solely on the small set achieves competitive performance compared with a large dataset. In rehearsal-based continual learning, the coreset is typically used in the memory replay buffer to stand for representative samples in previous tasks, and the coreset selection procedure is typically formulated as a bilevel problem. However, the typical bilevel formulation for coreset selection explicitly performs optimization over discrete decision variables with greedy search, which is computationally expensive. Several works consider other formulations to address this issue, but they ignore the nested nature of bilevel optimization problems and may not solve the bilevel coreset selection problem accurately. To address these issues, we propose a new bilevel formulation, where the inner problem tries to find a model which minimizes the expected training error sampled from a given probability distribution, and the outer problem aims to learn the probability distribution with approximately $K$ (coreset size) nonzero entries such that learned model in the inner problem minimizes the training error over the whole data. To ensure the learned probability has approximately $K$ nonzero entries, we introduce a novel regularizer based on the smoothed top-$K$ loss in the upper problem. We design a new optimization algorithm that provably converges to the $\epsilon$-stationary point with $O(1/\epsilon^4)$ computational complexity. We conduct extensive experiments in various settings in continual learning, including balanced data, imbalanced data, and label noise, to show that our proposed formulation and new algorithm significantly outperform competitive baselines. From bilevel optimization point of view, our algorithm significantly improves the vanilla greedy coreset selection method in terms of running time on continual learning benchmark datasets.

### Instruction
#### Prerequisites
```
$ pip install -r requirements.txt
```
Before running the code, please download the corresponding dataset and put it the data directory.
#### Example for running BCSR on Split CIFAR-100 

```
$ python main.py --select_type bcsr
```

### Citation
If you found this repository helpful, please cite our paper:
```
@inproceedings{
anonymous2023bilevel,
title={Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm},
author={Jie Hao, Kaiyi Ji, Mingrui Liu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=2dtU9ZbgSN}
}'''