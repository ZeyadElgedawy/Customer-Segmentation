
# Customer Segmentation Using Clustering

## Project Overview

This project applies **unsupervised machine learning** techniques to perform **customer segmentation** on a retail dataset. By grouping customers based on purchasing behavior and demographic features, businesses can better understand target audiences, tailor marketing strategies, and improve customer retention.

The dataset used is `Mall_Customers.csv`, containing attributes such as:

* **Age**
* **Annual Income (k\$)**
* **Spending Score (1–100)**

The analysis includes:

1. **Data exploration and visualization**
2. **Feature scaling**
3. **Clustering with K-Means**
4. **Clustering with DBSCAN**
5. **Cluster evaluation using Silhouette Score**
6. **Comparative insights from clustering results**

---

## Data Preprocessing

### Steps:

1. **Data Cleaning**

   * Removed the `CustomerID` column as it is not relevant for clustering.
   * Checked for missing values and duplicates.

2. **Exploratory Data Analysis (EDA)**

   * Visualized distributions for Age, Annual Income, and Spending Score.
   * Created boxplots to detect outliers.
   * Generated pair plots to identify relationships between features.

3. **Feature Scaling**

   * Standardized numerical features (`Annual Income (k$)` and `Spending Score (1–100)`) using **StandardScaler**.
   * Scaling ensures that all features contribute equally to the clustering process.

---

## K-Means Clustering

### Method:

* Iterated over **k values (2–12)**.
* Recorded **inertia** (elbow method) and **Silhouette Score** to determine the optimal number of clusters.
* Selected **k = 5** based on the highest silhouette score and elbow curve analysis.

### Results:

* Plotted **K-Means clusters** with centroids.
* Converted centroids back to the original scale for interpretability.
* Identified spending behavior per cluster.

---

## DBSCAN Clustering

### Method:

* Explored combinations of **eps** (0.01–1.0) and **min\_samples** (2–20).
* Used `get_scores_and_labels()` function to compute silhouette scores for each combination.
* Selected the best parameters based on silhouette performance.

### Results:

* Visualized DBSCAN clusters.
* Compared **average spending per cluster** with K-Means results.
* DBSCAN handled non-spherical cluster shapes better but sometimes generated noise points.

---

## Evaluation

* **K-Means Silhouette Score:** \~`best_kmeans_ss`
* **DBSCAN Silhouette Score:** \~`best_dict['best_score']`
* **Key Observations:**

  * K-Means provided well-separated, interpretable clusters for this dataset.
  * DBSCAN was more sensitive to parameter choices and worked well for non-linear structures.

---

## Future Improvements

* **Feature Expansion:** Include more variables such as purchase frequency or product categories for richer segmentation.
* **Dimensionality Reduction:** Apply PCA or t-SNE to visualize high-dimensional customer data.
* **Automated Hyperparameter Tuning:** Use grid search or Bayesian optimization for clustering parameters.
* **Hybrid Clustering Models:** Combine K-Means with hierarchical clustering for better initial centroid placement.
* **Time-based Segmentation:** Integrate temporal purchase behavior for dynamic segmentation.
