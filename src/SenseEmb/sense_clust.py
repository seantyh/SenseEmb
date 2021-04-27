from .sense_keyvec import SenseData
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import umap
import hdbscan

def make_hclust(sense_data: SenseData):
    vecs = sense_data.sense_vecs
    slabels = sense_data.sense_labels
    sns.clustermap(vecs.dot(vecs.T), 
        metric="cosine", method="complete", 
        cmap="summer", yticklabels=slabels)
    plt.gcf().set_figwidth(20)

def auto_clust(sense_data:SenseData, umap_n_neighbors=2, umap_ndim=2, 
                umap_min_dist=0.1, clust_min_samples=2, print_clust=False):
    vecs = sense_data.sense_vecs

    umap_inst = umap.UMAP(n_components=umap_ndim, 
                n_neighbors=umap_n_neighbors, metric='cosine', min_dist=umap_min_dist, random_state=4422)
    proj = umap_inst.fit_transform(vecs)
    clust = hdbscan.HDBSCAN(min_cluster_size=2, 
            min_samples=clust_min_samples, prediction_data=True).fit(proj)

    clabels = clust.labels_
    probs = clust.probabilities_
    sfreqs = np.array(sense_data.sense_freqs)
    slabels = sense_data.sense_labels

    if np.all(clabels < 0):
        prob_mat = np.array([])
        clust_freq = np.array([])
    else:
        prob_mat = hdbscan.all_points_membership_vectors(clust)
        clust_freq = (sfreqs[:, np.newaxis] * prob_mat).sum(0)

    sense_clusters = {}
    for clust_idx in np.unique(clabels):
        idx_list = (clabels==clust_idx).nonzero()[0]
        idx_list = sorted(idx_list, key=lambda x: -probs[x])
        
        sense_clusters[clust_idx] = [
            (i, probs[i], sfreqs[i], slabels[i]) 
            for i in idx_list
        ]

        if print_clust:
            print("-- Cluster %d --" % (clust_idx,))    
            print("\n".join(
                f"[{x[0]:2d}] {x[1]:.2f}({x[2]:3d}): {x[3]}" 
                for x in sense_clusters[clust_idx]))
            print("\n")

    if print_clust:
        if proj.shape[1] == 1:
            plt.scatter(proj[:,0], np.ones(proj.shape[0]), c=clust.labels_, cmap="Set1")
        else:
            plt.scatter(proj[:,0], proj[:,1], c=clust.labels_, cmap="Set1")

    return {
        "projection": proj,
        "sense_clusters": sense_clusters, 
        "sense_freqs": sfreqs, 
        "cluster_freqs": clust_freq, 
        "memberships": prob_mat }