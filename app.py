import streamlit as st
from image_segmentation import ImageSegmentationApp

def main():
    st.title("Image Segmentation App")
    app = ImageSegmentationApp()

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = app.load_image(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Sidebar for parameters
        st.sidebar.header("Segmentation Parameters")
        algorithm = st.sidebar.selectbox(
            "Choose Algorithm",
            ["K-Means", "Gaussian Mixture Model",  
             "Agglomerative Clustering", "BIRCH", "Mean Shift", "OPTICS", 
              "Affinity Propagation"]
        )
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
        color_palette = st.sidebar.selectbox("Color Palette", [ "viridis", "plasma", "inferno", "magma", "cividis"])
        max_size = st.sidebar.slider("Resize Image (Max Dimension)", 128, 1024, 512, step=128)

        
        use_pca = st.sidebar.checkbox("Use PCA ")

        # Preprocess image (resize)
        image = app.preprocess_image(max_size)  # Store the resized image

        # Apply selected algorithm with error handling
        try:
            if algorithm == "K-Means":
                segmented_image = app.apply_kmeans(n_clusters, color_palette, use_pca)
            elif algorithm == "Gaussian Mixture Model":
                segmented_image = app.apply_gmm(n_clusters, color_palette, use_pca)
            elif algorithm == "Agglomerative Clustering":
                segmented_image = app.apply_agglomerative(n_clusters, color_palette, use_pca)
            elif algorithm == "BIRCH":
                segmented_image = app.apply_birch(n_clusters, color_palette, use_pca)
            elif algorithm == "Mean Shift":
                segmented_image = app.apply_meanshift(color_palette, use_pca)
            elif algorithm == "OPTICS":
                segmented_image = app.apply_optics(color_palette, use_pca)
            elif algorithm == "Affinity Propagation":
                segmented_image = app.apply_affinity_propagation(color_palette, use_pca)
            
            # Display segmented image with optimized width
            st.image(segmented_image, caption=f"Segmented Image using {algorithm}", width=512)
        except Exception as e:
            st.error(f"Error during segmentation: {e}")


if __name__ == "__main__":
    main()