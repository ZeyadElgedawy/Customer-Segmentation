import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from utils import get_scores_and_labels
import numpy as np
import itertools

df = pd.read_csv("../data/Mall_Customers.csv")
print("\nData Frame Head:")
print(df.head())
print("\nData Frame info:")
print(df.info())

print("\nData Frame Description (Numerical Columns):")
print(df.describe())


missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values)

duplicated = df.duplicated().sum()
print("\nNumber of duplicated rows:",duplicated)

df = df.drop('CustomerID' , axis = 1)
sns.kdeplot(df['Annual Income (k$)'],fill = True)
plt.title("Annual Income distribution")
plots_save_path = r"..\outputs\Annual Income distribution.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.kdeplot(df['Age'], fill = True)
plt.title("Age distribution")
plots_save_path = r"..\outputs\Age distribution.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.kdeplot(df['Spending Score (1-100)'], fill = True)
plt.title("Spending Score (1-100) distribution")
plots_save_path = r"..\outputs\Spending Score (1-100) distribution.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.boxplot(df , x = df['Annual Income (k$)'])
plt.title('Boxplot of Annual Income (k$)')
plots_save_path = r"..\outputs\Boxplot of Annual Income (k$).png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.boxplot(df , x = df['Age'])
plt.title('Boxplot of Age')
plots_save_path = r"..\outputs\Boxplot of Age.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.boxplot(df , x = df['Spending Score (1-100)'])
plt.title('Boxplot of Spending Score (1-100)')
plots_save_path = r"..\outputs\Boxplot of Spending Score (1-100).png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.pairplot(df)
plt.suptitle('Mall_Customers pair plot', y=1.02)
plots_save_path = r"..\outputs\Mall_Customers pair plot.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[['Annual Income (k$)' , 'Spending Score (1-100)']])

scaled_data_df = pd.DataFrame(
    scaled_data,
    index = df.index,
    columns = ['Annual Income (k$)' , 'Spending Score (1-100)']
)

sns.kdeplot(scaled_data_df['Annual Income (k$)'],fill = True)
plt.title("Annual Income distribution after scaling")
plots_save_path = r"..\outputs\Annual Income distribution after scaling.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 

sns.kdeplot(scaled_data_df['Spending Score (1-100)'], fill = True)
plt.title("Spending Score (1-100) distribution after scaling")
plots_save_path = r"..\outputs\Spending Score (1-100) distribution after scaling.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()
 
####################### K-Means clustering #######################
inertia = []
silhouette_scores = []
k_values = range(2 , 13)
for k in k_values:
    kmeans = KMeans(n_clusters = k , random_state = 42 , max_iter = 1000)

    clusters_labels = kmeans.fit_predict(scaled_data_df)

    sil_score = silhouette_score(scaled_data_df,clusters_labels)

    silhouette_scores.append(sil_score)

    inertia.append(kmeans.inertia_)

best_kmeans_ss = np.max(silhouette_scores)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('KMeans Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)

plots_save_path = r"..\outputs\KMeans Inertia & Silhouette Scores for Different Values of k.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
 

kmeans = KMeans(n_clusters=5, random_state=42, max_iter=1000)

cluster_labels = kmeans.fit_predict(scaled_data_df)

df['Clusters'] = cluster_labels

centroids = kmeans.cluster_centers_

centroids_original = scaler.inverse_transform(centroids)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Clusters',
    palette='tab10',
    s=100
)


plt.scatter(
    centroids_original[:,0],
    centroids_original[:,1],
    marker='X', 
    s=200,      
    c='black', 
    edgecolors='w', 
    label='Centroids'
)

plt.title('K-Means Clustering with 5 Clusters and Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)

plots_save_path = r"..\outputs\KMeans_Clusters_Plot_with_Centroids.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

####################### DBSCAN clustering #######################
epsilons = np.linspace(0.01 , 1 , num = 15)

minPts = np.arange(2 , 20 , step = 3 )

combinations = list(itertools.product(epsilons , minPts))
print(f"\nLengh of combinations: {len(combinations)}")

best_dict = get_scores_and_labels(combinations , scaled_data_df)

print("\nBest Dictionary:")
print(best_dict)

df2 = df.copy()

df2['Clusters'] = best_dict['best_labels']

print("\nClusters value counts:")
print(df2['Clusters'].value_counts())

sns.scatterplot(
    data=df2,
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Clusters',
    palette='tab10',
    s=100
)
plt.title('DBSCAN Clustering with 5 Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plots_save_path = r"..\outputs\DBSCAN Clusters.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"\nK-Means clustering silhouette score: {best_kmeans_ss:.3f}")
print(f"\nDSCAN clustering silhouette score: {best_dict['best_score']:.3f}")

avg_spending_kmean = df.groupby('Clusters')['Spending Score (1-100)'].mean()
print("\nAverage spending per cluster based on K-Means clustering")
print(avg_spending_kmean)

avg_spending_dbscan = df2.groupby('Clusters')['Spending Score (1-100)'].mean()
print("\nAverage spending per cluster based on DBSCAN clustering")
print(avg_spending_dbscan)