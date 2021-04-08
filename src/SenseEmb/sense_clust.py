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

def auto_clust(sense_data:SenseData, clust_min_samples=2):
    vecs = sense_data.sense_vecs

    umap_inst = umap.UMAP(n_components=2, 
                n_neighbors=2, metric='cosine', min_dist=0.1, random_state=4422)
    proj = umap_inst.fit_transform(vecs)
    clust = hdbscan.HDBSCAN(min_cluster_size=2, 
            min_samples=clust_min_samples).fit(proj)
    plt.scatter(proj[:,0], proj[:,1], c=clust.labels_, cmap="Set1")

    clabels = clust.labels_
    probs = clust.probabilities_
    sfreqs = sense_data.sense_freqs
    slabels = sense_data.sense_labels

    for clust_idx in np.unique(clabels):
        idx_list = (clabels==clust_idx).nonzero()[0]
        idx_list = sorted(idx_list, key=lambda x: -probs[x])
        
        print("-- Cluster %d --" % (clust_idx,))    
        print("\n".join(f"{probs[i]:.2f}({sfreqs[i]:3d}): {slabels[i]}" for i in idx_list))
        print("\n")