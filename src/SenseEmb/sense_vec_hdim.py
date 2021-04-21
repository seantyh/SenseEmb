import umap
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import numpy as np
import numpy.linalg as la
import igraph as ig
from tqdm.auto import tqdm
import json
from scipy.spatial import Delaunay, ConvexHull, Voronoi
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

def compute_cut_points(svecs, thresholds=.9, min_sv=1, max_sv=4):
    singular_vals = la.svd(svecs)[1]
    frob_norm = np.sqrt((singular_vals**2).sum())
    vars_explained = np.sqrt((singular_vals**2).cumsum())/frob_norm
    sv_cut = np.argmax(vars_explained > thresholds)
    sv_cut = min(max(min_sv, sv_cut), max_sv)
    var_ratio = vars_explained[sv_cut]
    return sv_cut, var_ratio

def compute_geometry(vecs):    
    hull = ConvexHull(vecs)
    vor = Voronoi(vecs)    
    sense_vols = np.zeros(vor.npoints)
    in_open_region = []
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:
            sense_vols[i] = np.inf
            in_open_region.append(i)
        else:
            sense_vols[i] = ConvexHull(vor.vertices[indices]).volume
    all_regions_volume = sense_vols[np.isfinite(sense_vols)].sum()
    return dict(
        hull=hull, voronoi=vor,
        sense_vols=sense_vols, hull_vol=hull.volume,
        in_open_region=in_open_region,
        all_regions_volume=all_regions_volume
    )        

def compute_community(low_vecs, 
        sense_labels, sense_ids, sense_vols, all_regions_volume):    

    n_neigh = int(np.round(np.sqrt(low_vecs.shape[0])))
    uinst = umap.UMAP(n_components=2, n_neighbors=n_neigh, 
            min_dist=0.0, transform_mode="graph", metric="cosine")
    uinst.fit(low_vecs)
    graph = uinst.graph_            
    g2 = ig.Graph.Weighted_Adjacency(graph.todense(), mode="undirected")
    vcs = g2.community_leading_eigenvector(weights="weight")    

    memberships = vcs.membership
    clusters = []
    sense_idxs = np.arange(len(memberships))
    for clust_idx in np.unique(memberships):        
        vols = sense_vols[memberships == clust_idx]
        volume_clust = vols[np.isfinite(vols)].sum()
        vol_ratio = volume_clust/all_regions_volume
                
        clust_senses = []
        for sidx in sense_idxs[memberships == clust_idx]:            
            clust_senses.append((sense_ids[sidx], sense_labels[sidx]))        
        cluster_x = dict(vol_ratio=vol_ratio, senses=clust_senses)
        clusters.append(cluster_x)
    return dict(clutsers=clusters)
    
def compute_highdim_indices(word, skv):
    try:
        sdata = skv.make_sense_vectors(word)
    except Exception as ex:
        return {}

    svecs = sdata.sense_vecs
    if svecs.shape[0] < 3:
        return {}

    sv_cut, var_ratio = compute_cut_points(svecs, thresholds=0.9, min_sv=1, max_sv=4)
    pca_svecs = PCA(sv_cut+1).fit_transform(svecs)
    
    print(word, pca_svecs.shape)
    try:
        geo_indices = compute_geometry(pca_svecs)    
    except Exception as ex:
        print("Exception when computing geometry of word ", word)
        print(ex)
        geo_indices = {}

    slabels = [f"{i}. {x}" for i, x in enumerate(sdata.sense_labels)]
    comm_indices = {}
    try:        
        if geo_indices and pca_svecs.shape[0] > pca_svecs.shape[1]:
            comm_indices = compute_community(pca_svecs, 
                slabels, sdata.sense_ids, 
                geo_indices["sense_vols"], geo_indices["all_regions_volume"])        
    except Exception as ex:
        print("Exceptino when computing community of word ", word)
        print(ex)        

    return {
        "word": word,
        "n_sense": svecs.shape[0], 
        "sense_freqs": sdata.sense_freqs,
        "pca_dim": sv_cut+1, 
        "sv_cut": sv_cut, "sv_ratio": var_ratio,
        **geo_indices, **comm_indices}

