import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap
from sklearn.preprocessing import StandardScaler
import struct


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(size, rows * cols)


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def apply_dimension_reduction(X_train, X_test, n_components=50, method='pca'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'kpca':
        reducer = KernelPCA(n_components=n_components, kernel='rbf', random_state=42)
    elif method == 'mds':
        reducer = MDS(n_components=n_components, random_state=42, n_jobs=-1)
    elif method == 'lle':
        reducer = LocallyLinearEmbedding(n_components=n_components, random_state=42, n_jobs=-1)
    elif method == 'isomap':
        reducer = Isomap(n_components=n_components, n_jobs=-1)
    else:
        raise ValueError("Unknown method")

    X_train_reduced = reducer.fit_transform(X_train_scaled)
    if hasattr(reducer, 'transform'):
        X_test_reduced = reducer.transform(X_test_scaled)
    else:
        X_test_reduced = reducer.fit_transform(X_test_scaled)

    return X_train_reduced, X_test_reduced