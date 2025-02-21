import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch, DBSCAN, MeanShift, OPTICS,  AffinityPropagation
from sklearn.mixture import GaussianMixture
from PIL import Image
from image_processing import resize_image
from sklearn.decomposition import PCA


class ImageSegmentationApp:
    def __init__(self):
        self.image = None
        self.segmented_image = None

    def load_image(self, uploaded_file):
        """Load an image from an uploaded file."""
        self.image = np.array(Image.open(uploaded_file))
        return self.image

    def preprocess_image(self, max_size=1024):
        """Resize the image to reduce its dimensions for faster processing."""
        self.image = resize_image(self.image, max_size)
        return self.image

    def _apply_pca(self, pixels, use_pca):
        """Apply PCA if the checkbox is selected."""
        if use_pca:
            pca = PCA(n_components=2)  # Reduce to 2 dimensions
            pixels = pca.fit_transform(pixels)
        return pixels

    def _apply_color_palette(self, labels, color_palette, n_clusters):
        """Apply the selected color palette to the segmented image."""
        if color_palette == 'default':
            colors = plt.cm.get_cmap('viridis', n_clusters)(np.linspace(0, 1, n_clusters))[:, :3] * 255
        else:
            colors = plt.cm.get_cmap(color_palette, n_clusters)(np.linspace(0, 1, n_clusters))[:, :3] * 255
        segmented_image = colors[labels].reshape(self.image.shape)
        return segmented_image.astype(np.uint8)

    def apply_kmeans(self, n_clusters, color_palette, use_pca=False):
        """Apply K-Means clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_gmm(self, n_clusters, color_palette, use_pca=False):
        """Apply Gaussian Mixture Model clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(pixels)
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_spectral_clustering(self, n_clusters, color_palette, use_pca=False):
        """Apply Spectral Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(pixels)
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_agglomerative(self, n_clusters, color_palette, use_pca=False):
        """Apply Agglomerative Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglo.fit_predict(pixels)
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_birch(self, n_clusters, color_palette, use_pca=False):
        """Apply BIRCH Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        birch = Birch(n_clusters=n_clusters)
        labels = birch.fit_predict(pixels)
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_meanshift(self, color_palette, use_pca=False):
        """Apply Mean Shift Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        meanshift = MeanShift()
        labels = meanshift.fit_predict(pixels)
        n_clusters = len(np.unique(labels))
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_optics(self, color_palette, use_pca=False):
        """Apply OPTICS Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        optics = OPTICS()
        labels = optics.fit_predict(pixels)
        n_clusters = len(np.unique(labels))
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image

    def apply_affinity_propagation(self, color_palette, use_pca=False):
        """Apply Affinity Propagation Clustering to the image."""
        pixels = self.image.reshape(-1, 3)
        pixels = self._apply_pca(pixels, use_pca)
        affinity_propagation = AffinityPropagation()
        labels = affinity_propagation.fit_predict(pixels)
        n_clusters = len(np.unique(labels))
        self.segmented_image = self._apply_color_palette(labels, color_palette, n_clusters)
        return self.segmented_image