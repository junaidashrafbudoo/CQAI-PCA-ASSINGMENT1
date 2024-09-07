import streamlit as st
import cv2
import os
from sklearn.decomposition import PCA, KernelPCA

# Cache the directory listing to avoid repeated I/O operations
@st.cache_data
def get_image_list(directory="Image"):
    return os.listdir(directory)

# Cache the image reading to avoid repeated decoding
@st.cache_data
def load_image(file_path):
    return cv2.imread(file_path)

# Function to perform PCA and inverse transform
def apply_pca(channel, components):
    try:
        pca = PCA(n_components=components)
        transformed = pca.fit_transform(channel)
        return pca.inverse_transform(transformed)
    except Exception as e:
        st.error(f"Error applying PCA: {e}")
        return channel  # Return the original channel if PCA fails

def apply_kernel_pca(channel, components, kernel="rbf"):
    try:
        kpca = KernelPCA(n_components=components, kernel=kernel, fit_inverse_transform=True)
        transformed = kpca.fit_transform(channel)
        return kpca.inverse_transform(transformed)
    except Exception as e:
        st.error(f"Error applying Kernel PCA: {e}")
        return channel  # Return the original channel if Kernel PCA fails

# Utility function to convert an image to PNG format for downloading
def convert_to_png(image):
    _, buffer = cv2.imencode('.png', image)
    return buffer.tobytes()
