# Neural Operators for Partial Differential Equations (PDEs)
Partial differential equations (PDEs) play a dominant role in the mathematical modeling of many complex dynamical processes. Solving these PDEs often requires prohibitively high computational costs, especially when multiple evaluations must be made for different parameters or conditions. After training, neural operators can provide PDEs solutions significantly faster than traditional PDE solvers. In this work, invariance properties and computational complexity of two neural operators are examined for transport PDE of a scalar quantity. Neural operator based on [graph kernel network (GKN)](https://arxiv.org/abs/2003.03485) operates on graph-structured data to incorporate nonlocal dependencies. Here we propose a modified formulation of GKN to achieve frame invariance. [Vector cloud neural network (VCNN)](https://arxiv.org/abs/2103.06685) is an alternate neural operator with embedded frame invariance which operates on point cloud data. GKN-based neural operator demonstrates slightly better predictive performance compared to VCNN. However, GKN requires an excessively high computational cost that increases quadratically with the increasing number of discretized objects as compared to a linear increase for VCNN.

### Schematic comparison of GKN and VCNN
<div align=center><img width="966" src="https://github.com/muhammadirfanzafar/pde_neural_operators/blob/main/figures/schematic_gkn_vcnn.png"/></div>

This repository contains the code for the following paper:
* MI Zafar, J Han, XH Zhou, and H Xiao. Frame invariance and scalability of neural operators for partial differential equations. *Communications in Computational Physics*, Accepted for publication. Also available at arXiv: [https://arxiv.org/abs/2112.14769](https://arxiv.org/abs/2112.14769)

### Requirements
* PyTorch
* PyTorch Geometric
* pandas
* sklearn

### Files
The code is in the form of simple python scripts. Each script shall be directly runnable.

### Datasets
We provide the dataset for a flow over a symmetric NACA-0012 airfoil in a two-dimensional space. The data generation can be found in the paper. Data is stored as tensors in .pt file extension using PyTorch. Once downloaded from the link below and saved in 'data' folder, the scripts should be runnable.

[Airfoil dataset](https://drive.google.com/drive/folders/1nZRlF8eFbD76DAvH5Q-03iVUROIKwl7e?usp=sharing)

Contributors:
-------------
* Muhammad-Irfan Zafar
* Jiequn Han
* Xu-Hui Zhou
* Heng Xiao

Contact: Muhammad-Irfan Zafar     
Email address: zmuhammadirfan@vt.edu
