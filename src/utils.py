
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np


def get_scores_and_labels(combinations , X):
    scores = []
    all_labels_list = []

    for i , (eps , minPts) in enumerate(combinations):
        dbScan = DBSCAN(eps = eps , min_samples = minPts).fit(X)
        labels = dbScan.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)

        if -1 in labels_set:
            num_clusters -=1
        
        if (num_clusters < 2) or (num_clusters > 50):
            scores.append(-1)
            all_labels_list.append('Ignored')
            c = (eps , minPts)
            print(f"\nCombination {c} on iteration {i + 1} of {len(combinations)} has {num_clusters} clusters (Ignored).")
            continue
        
        scores.append(silhouette_score(X , labels))
        all_labels_list.append(labels)

        print(f"\nIndex {i} , score : {scores[-1]} , Labels : {all_labels_list[-1]} , NumClusters : {num_clusters}")

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]

    return {
        'best_epsilon' : best_parameters[0],
        'best_min_samples' : best_parameters[1],
        'best_score' : best_score,
        'best_labels' : best_labels,
    }