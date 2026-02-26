import joblib

scaler = joblib.load('reports/models/scaler.pkl')
pca = joblib.load('reports/models/pca.pkl')
kmeans = joblib.load('reports/models/kmeans.pkl')
dbscan = joblib.load('reports/models/dbscan.pkl')

print('Scaler:')
print(f'  mean_ shape: {scaler.mean_.shape}')
print(f'  scale_ shape: {scaler.scale_.shape}')

print('\nPCA:')
print(f'  n_components: {pca.n_components_}')
print(f'  explained_variance_ratio_: {pca.explained_variance_ratio_}')

print('\nKMeans:')
print(f'  n_clusters: {kmeans.n_clusters}')
print(f'  inertia: {kmeans.inertia_}')

print('\nDBSCAN:')
print(f'  eps: {dbscan.eps}')
print(f'  min_samples: {dbscan.min_samples}')
