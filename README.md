# SECE
Spatial region-related embedding and Cell type-related embedding of spatial transcriptomics.

Spatially resolved transcriptomics sequencing (ST-seq) can profile gene expression while preserving spatial location in tissue microenvironment. However, learning effective low dimensional embedding of ST-seq is hindered by insufficient use of spatial location and high level of technical noise. To this end, we propose spatial region-related embedding and cell type-related embedding learning method (SECE), a framework for spatial region-related embedding (SE) learning and cell type-related embedding (CE) learning. SECE uses Autoencoder with negative binomial distribution to model expression counts and learn CE, followed by Graph Attention network to learn SE using adjacency matrix and similarity matrix constructed from spatial location. By applying SECE to diverse ST-seq data with different resolutions and different tissue types, we obtained state-of-the-art spatial domain identification results and demonstrated that SE can be used for tasks such as visualization and trajectory inference.

![Fig2](https://user-images.githubusercontent.com/53144397/191158987-a855f2a8-cf23-4930-a209-079b9a6d1f8c.png)


# Installation

Before installing SECE, please ensure that the software dependencies are installed.

```python
   scanpy==1.9.1
   pytorch==1.4.0
   torch-geometric==2.0.4
   torch-cluster==1.5.4
   torch-scatter==2.0.3 
   torch-sparse==0.6.1
   torch-spline-conv==1.2.0
   torchvision==0.5.0
```
SECE can be downloaded via pip:

```python

   pip install SECE
```
Then SECE can be used in python:

```python
   import SECE
```
# Tutorial

# Citation
