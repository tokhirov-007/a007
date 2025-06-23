from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Сгенерируем данные
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Обучаем
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Визуализация
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("✅ KMeans Clustering (Unsupervised)")
plt.savefig("kmeans_plot.png")

