# SECE
Spatial region-related embedding and Cell type-related embedding of spatial transcriptomics.

Spatially resolved transcriptomics sequencing (ST-seq) can profile gene expression while preserving spatial location in tissue microenvironment. However, learning effective low dimensional embedding of ST-seq is hindered by insufficient use of spatial location and high level of technical noise. To this end, we propose spatial region-related embedding and cell type-related embedding learning method (SECE), a framework for spatial region-related embedding (SE) learning and cell type-related embedding (CE) learning. SECE uses Autoencoder with negative binomial distribution to model expression counts and learn CE, followed by Graph Attention network to learn SE using adjacency matrix and similarity matrix constructed from spatial location. By applying SECE to diverse ST-seq data with different resolutions and different tissue types, we obtained state-of-the-art spatial domain identification results and demonstrated that SE can be used for tasks such as visualization and trajectory inference.
![image](https://user-images.githubusercontent.com/53144397/191151260-addf53fd-7142-44de-a06c-b3dc1e73be53.png)

# Installation
pip install SECE

# Tutorial

# Citation
