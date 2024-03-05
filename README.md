# SECE
Spatial region-related embedding and Cell type-related embedding of spatial transcriptomics.

SECE is an accurate spatial domain identification method for ST data. In contrast to the existing methods, SECE introduces global spatial proximity to weaken the influence of local expression proximity on spatial domain identification results. We compared SECE to three state-of-the-art approaches. SECE consistently outperforms them in various tasks including spatial domain and cell type identification, low-dimensional visualization, and trajectory inference, revealing many novel biological insights. SECE is versatile on datasets from different ST platforms, for both high-resolution platforms, such as STARmap, Slide-seqV2 and Stereo-seq, and relatively low-resolution ones such as Visium. For ease of use, SECE was implemented and integrated into the workflow of ST analysis using the Anndata data structure, which can be used in the SCANPY package for visualization and downstream analyses. The capability of highly accurate identification of spatial domains and cell types enables SECE to facilitate the discovery of biological insights from ST data.

<img width="416" alt="image" src="https://github.com/yuyuanyuana/SECE/assets/53144397/33a1708c-7255-4fe8-9bbc-f882b453ab09">

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
