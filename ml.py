import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Config
# ---------------------------
DEFAULT_DATA_PATH = r"C:\\Users\\dhrit\\Downloads\\online+retail\\Online Retail.xlsx"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
MODEL_DIR = os.path.join(REPORTS_DIR, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Load
# ---------------------------
print(f"Loading data from: {DATA_PATH}")
xl = pd.ExcelFile(DATA_PATH)
print("Sheets:", xl.sheet_names)
df = pd.read_excel(DATA_PATH, sheet_name=xl.sheet_names[0])
print("Raw shape:", df.shape)

# ---------------------------
# Clean
# ---------------------------
# 1) Drop rows without CustomerID (cannot assign to a customer)
# 2) Remove cancelled invoices (InvoiceNo starts with 'C')
# 3) Keep only positive Quantity and UnitPrice

df = df.copy()
if "CustomerID" in df.columns:
    df = df.dropna(subset=["CustomerID"])

if "InvoiceNo" in df.columns:
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

if "Quantity" in df.columns:
    df = df[df["Quantity"] > 0]

if "UnitPrice" in df.columns:
    df = df[df["UnitPrice"] > 0]

print("Clean shape:", df.shape)

# ---------------------------
# Feature Engineering (Customer-level)
# ---------------------------
# TotalPrice per row

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
max_date = df["InvoiceDate"].max()

agg = df.groupby("CustomerID").agg(
    last_purchase=("InvoiceDate", "max"),
    frequency=("InvoiceNo", "nunique"),
    monetary=("TotalPrice", "sum"),
    avg_basket=("TotalPrice", "mean"),
    unique_products=("StockCode", "nunique"),
)

agg["recency"] = (max_date - agg["last_purchase"]).dt.days

# final feature set
features = agg[["recency", "frequency", "monetary", "avg_basket", "unique_products"]].copy()

# ---------------------------
# Preprocess: log1p for skew + standardize
# ---------------------------
features_log = np.log1p(features)
scaler = StandardScaler()
X = scaler.fit_transform(features_log)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ---------------------------
# PCA (2D for visualization)
# ---------------------------

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))

# ---------------------------
# K-Means: find best k by silhouette
# ---------------------------
ks = list(range(2, 11))
km_inertia = []
km_sil = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    km_inertia.append(km.inertia_)
    try:
        km_sil.append(silhouette_score(X, labels))
    except Exception:
        km_sil.append(np.nan)

best_k = ks[int(np.nanargmax(km_sil))]

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
km_labels = kmeans.fit_predict(X)
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))

# ---------------------------
# DBSCAN: choose eps from 90th percentile of k-distance
# ---------------------------
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples)
nbrs.fit(X)
# distances to kth nearest neighbor
k_distances = np.sort(nbrs.kneighbors(X)[0][:, -1])
# simple heuristic for eps
eps = float(np.percentile(k_distances, 90))

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
db_labels = dbscan.fit_predict(X)
joblib.dump(dbscan, os.path.join(MODEL_DIR, "dbscan.pkl"))

# ---------------------------
# Metrics
# ---------------------------

def safe_silhouette(data, labels):
    # silhouette needs at least 2 clusters and no single-cluster result
    unique = set(labels)
    if len(unique) <= 1:
        return np.nan
    # all points labeled as noise
    if unique == {-1}:
        return np.nan
    # if only noise + one cluster, silhouette is not meaningful
    if len(unique - {-1}) <= 1:
        return np.nan
    return silhouette_score(data, labels)

km_sil_best = safe_silhouette(X, km_labels)

# DBSCAN cluster count (exclude noise)
clusters_db = len(set(db_labels) - {-1})

metrics = pd.DataFrame([
    {
        "algorithm": "KMeans",
        "n_clusters": int(best_k),
        "silhouette": float(km_sil_best) if km_sil_best == km_sil_best else np.nan,
        "params": json.dumps({"n_clusters": int(best_k)})
    },
    {
        "algorithm": "DBSCAN",
        "n_clusters": int(clusters_db),
        "silhouette": float(safe_silhouette(X, db_labels)) if safe_silhouette(X, db_labels) == safe_silhouette(X, db_labels) else np.nan,
        "params": json.dumps({"eps": eps, "min_samples": min_samples})
    }
])

metrics_path = os.path.join(REPORTS_DIR, "metrics.csv")
metrics.to_csv(metrics_path, index=False)

# ---------------------------
# Save labeled customer table
# ---------------------------
output = features.copy()
output["kmeans_cluster"] = km_labels
output["dbscan_cluster"] = db_labels
output.to_csv(os.path.join(REPORTS_DIR, "customer_clusters.csv"))

# ---------------------------
# Cluster Profiles
# ---------------------------
# Mean RFM per cluster to describe customer types

kmeans_profile = (
    output.groupby("kmeans_cluster")[["recency", "frequency", "monetary"]]
    .mean()
    .sort_index()
)
kmeans_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles_kmeans.csv"))

dbscan_profile = (
    output.groupby("dbscan_cluster")[["recency", "frequency", "monetary"]]
    .mean()
    .sort_index()
)
dbscan_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles_dbscan.csv"))

# ---------------------------
# Plots
# ---------------------------
# Elbow
plt.figure(figsize=(6, 4))
plt.plot(ks, km_inertia, marker="o")
plt.title("K-Means Elbow")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_elbow.png"), dpi=150)
plt.close()

# Silhouette vs k
plt.figure(figsize=(6, 4))
plt.plot(ks, km_sil, marker="o")
plt.title("K-Means Silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_silhouette.png"), dpi=150)
plt.close()

# PCA scatter for KMeans
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, s=8, cmap="tab10")
plt.title(f"K-Means Clusters (k={best_k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_kmeans.png"), dpi=150)
plt.close()

# k-distance plot for DBSCAN
plt.figure(figsize=(6, 4))
plt.plot(k_distances)
plt.title("DBSCAN k-distance (k=5)")
plt.xlabel("Points sorted by distance")
plt.ylabel("k-distance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dbscan_kdistance.png"), dpi=150)
plt.close()

# PCA scatter for DBSCAN
plt.figure(figsize=(6, 5))
mask_noise = db_labels == -1
plt.scatter(
    X_pca[~mask_noise, 0],
    X_pca[~mask_noise, 1],
    c=db_labels[~mask_noise],
    s=8,
    cmap="tab10",
)
plt.scatter(
    X_pca[mask_noise, 0],
    X_pca[mask_noise, 1],
    c="#999999",
    s=8,
)
plt.title("DBSCAN Clusters (noise in gray)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_dbscan.png"), dpi=150)
plt.close()

print("Done. Outputs saved in:", REPORTS_DIR)
print("Figures saved in:", FIG_DIR)
print("Models saved in:", MODEL_DIR)
print("Metrics:")
print(metrics)
