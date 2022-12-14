# SECE
Spatial region-related embedding and Cell type-related embedding of spatial transcriptomics.

Spatially resolved transcriptomics sequencing (ST-seq) can profile gene expression while preserving spatial location in tissue microenvironment. However, learning effective low dimensional embedding of ST-seq is hindered by insufficient use of spatial location and high level of technical noise. To this end, we propose spatial region-related embedding and cell type-related embedding learning method (SECE), a framework for spatial region-related embedding (SE) learning and cell type-related embedding (CE) learning. SECE uses Autoencoder with negative binomial distribution to model expression counts and learn CE, followed by Graph Attention network to learn SE using adjacency matrix and similarity matrix constructed from spatial location. By applying SECE to diverse ST-seq data with different resolutions and different tissue types, we obtained state-of-the-art spatial domain identification results and demonstrated that SE can be used for tasks such as visualization and trajectory inference.

![fig1](https://user-images.githubusercontent.com/53144397/195017747-f990e641-3568-45c6-bf33-a46a6c7875cf.png)

## Installation

Before installing SECE, please ensure that the software dependencies are installed.

```python
   scanpy
   pytorch
   torch-geometric
```
SECE can be downloaded via pip:

```python
   pip install SECE
```
Then SECE can be used in python:

```python
   import SECE
```
## Tutorial

Tutorials and installation guides are available in
    the [documentation](https://sece-tutorial.readthedocs.io/en/latest/index.html).

## Citation
