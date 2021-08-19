# Unsupervised Differentiable Multi-aspect Network Embedding (asp2vec)

<p align="center">
    <a href="https://www.kdd.org/kdd2020/" alt="Conference">
        <img src="https://img.shields.io/badge/KDD'20-brightgreen" /></a>   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>   
</p>

### Overview
Network embedding is an influential graph mining technique for representing nodes in a graph as distributed vectors. However, the majority of network embedding methods focus on learning a single vector representation for each node, which has been recently criticized for not being capable of modeling multiple aspects of a node. To capture the multiple aspects of each node, existing studies mainly rely on offline graph clustering performed prior to the actual embedding, which results in the cluster membership of each node (i.e., node aspect distribution) fixed throughout training of the embedding model. We argue that this not only makes each node always have the same aspect distribution regardless of its dynamic context, but also hinders the end-to-end training of the model that eventually leads to the final embedding quality largely dependent on the clustering. In this paper, we propose a novel end-to-end framework for multi-aspect network embedding, called asp2vec, in which the aspects of each node are dynamically assigned based on its local context. More precisely, among multiple aspects, we dynamically assign a single aspect to each node based on its current context, and our aspect selection module is end-to-end differentiable via the Gumbel-Softmax trick. We also introduce the aspect regularization framework to capture the interactions among the multiple aspects in terms of relatedness and diversity. We further demonstrate that our proposed framework can be readily extended to heterogeneous networks. Extensive experiments towards various downstream tasks on various types of homogeneous networks and a heterogeneous network demonstrate the superiority of asp2vec.

### Paper
- [ **Unsupervised Differentiable Multi-aspect Network Embedding (*KDD 2020*)** ](https://arxiv.org/abs/2006.04239)
  - [_**Chanyoung Park**_](http://pcy1302.github.io), Carl Yang, Qi Zhu, Donghyun Kim, Hwanjo Yu, Jiawei Han
  - [[Video(Long)]](https://youtu.be/DOdRmz1gLdc) [[Video(Short)]](https://www.youtube.com/watch?v=OksyuNXcI_c) [[Slides]](https://pcy1302.github.io/data/KDD20_slide.pdf)

### Requirements
- Python version: 3.6.8
- Pytorch version: 1.2.0
- fastrand (Fast random number generation in Python) (https://github.com/lemire/fastrand)
- scikit-learn


### How to Run
````
git clone https://github.com/pcy1302/asp2vec.git
cd asp2vec/
conda create -n py36 python=3.6.8
source activate py36
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
pip install fastrand
conda install scikit-learn
cd src/
````
### For instructions regarding data, please check [````data````](https://github.com/pcy1302/asp2vec/tree/master/data) directory

# Execute asp2vec on Filmtrust dataset
````
python main.py --embedder asp2vec --dataset filmtrust --isSoftmax --isGumbelSoftmax --dim 20 --num_aspects 5 --isReg --isInit
````


### Arguments
````--embedder````: name of the embedding method

````--dataset````: name of the dataset

````--isInit````: If ````True````, warm-up step is performed

````--iter_max````: maximum iteration

````--dim````: dimension size of a node

````--window_size````: window_size to determine the context

````--path_length````: lentgh of each random walk

````--num_neg````: number of negative samples

````--num_walks_per_node````: number of random walks starting from each node

````--lr````: learning rate

````--patience````: when to stop (early stop criterion)

````--isReg````: enable aspect regularization framework

````--reg_coef````: lambda in aspect regularization framework

````--threshold````: threshold for aspect regularization

````--isSoftmax````: enable softmax

````--isGumbelSoftmax````: enable gumbel-softmax

````--isNormalSoftmax````: enable conventional softmax

````--num_aspects````: number of predefined aspects (K)

### Reference
If you find ````asp2vec```` useful in your research, please cite the following paper:
- Park, Chanyoung; Yang, Carl; Zhu, Qi; Kim, Donghyun; Yu, Hwanjo; Han, Jiawei, "Unsupervised Differentiable Multi-aspect Network Embedding." KDD2020.
- Bibtex
````
@inproceedings{par2020unsup,
  title={Unsupervised Differentiable Multi-aspect Network Embedding},
  author={Park, Chanyoung and Yang, Carl and Zhu, Qi and Kim, Donghyun and Yu, Hwanjo and Han, Jiawei},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1435--1445},
  year={2020}
}
````
