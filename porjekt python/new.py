import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
file_path = "customer_segmentation.csv"
df = pd.read_csv(file_path)

# Basic Data Summary
print("Dataset Overview:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
plt.title("Income Distribution")
plt.show()

# Spending Score vs. Income
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Customer Category'])
plt.title("Spending Score vs. Income")
plt.show()

# Clustering (K-Means) for Customer Segmentation
features = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Purchase Frequency']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Finding optimal clusters using Elbow Method and Silhouette Score
inertia = []
silhouette_scores = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Clustering")
plt.show()

# Applying K-Means with chosen clusters (let's assume k=4 based on the elbow method)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing Clusters
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis')
plt.title("Customer Segments")
plt.show()

# Demographic Insights
plt.figure(figsize=(8,5))
sns.countplot(x=df['Gender'], hue=df['Customer Category'])
plt.title("Gender Distribution by Customer Category")
plt.show()

# Geographical Analysis
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Location'], y=df['Spending Score (1-100)'])
plt.xticks(rotation=45)
plt.title("Spending Score by Location")
plt.show()

# Saving the clustered dataset
df.to_csv("processed_customer_segmentation.csv", index=False)

# Display Cluster Statistics
cluster_summary = df.groupby('Cluster').agg({'Annual Income (k$)': ['mean', 'median'],
                                             'Spending Score (1-100)': ['mean', 'median'],
                                             'Purchase Frequency': ['mean', 'median']})
print("Cluster Summary:\n", cluster_summary)
