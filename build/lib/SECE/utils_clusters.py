import scanpy as sc
import numpy as np

def mclust_R(adata, num_cluster=10, modelNames='EEE', used_obsm='latent', random_seed=2022, key_add='clusters'):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_add] = mclust_res
    adata.obs[key_add] = adata.obs[key_add].astype('str')
    return adata



def cluster_func(adata, clustering, use_rep, res=1, cluster_num=None, key_add='cluster'):
    if clustering == 'louvain':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.louvain(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'leiden':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.leiden(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=cluster_num, random_state=2022).fit(adata.obsm[use_rep])
        adata.obs[key_add] = km.labels_
    if clustering == 'mclust':
        adata = mclust_R(adata, num_cluster=cluster_num, modelNames='EEE', used_obsm=use_rep, random_seed=2022, key_add=key_add)
    adata.obs[key_add] = adata.obs[key_add].astype('category')
    return adata

