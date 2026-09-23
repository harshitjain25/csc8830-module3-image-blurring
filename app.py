"""
CSC 8830 - Computer Vision
Module 3 Assignment

Web Application:
Image Blurring using Spatial and Fourier Domain Filtering

Run using:
streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
from PIL import Image

from image_filtering import (
    create_average_kernel,
    spatial_blur,
    fourier_blur,
    compare_results
)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CSC 8830 - Image Blurring",
    layout="wide"
)


# ---------------------------------------------------------
# Title and Description
# ---------------------------------------------------------

st.title("CSC 8830 - Image Blurring")
st.subheader("Spatial Domain vs Fourier Domain Filtering")

st.write(
    """
    This application demonstrates the Convolution Theorem.

    Image convolution in the spatial domain should produce
    the same result as multiplication in the Fourier domain.
    """
)


# ---------------------------------------------------------
# Interactive Demonstration
# ---------------------------------------------------------

st.header("Working Demonstration")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

kernel_size = st.select_slider(
    "Select Blur Kernel Size",
    options=[3, 5, 7, 9, 11],
    value=5
)


if uploaded_file is not None:

    # Load uploaded image
    original_image = Image.open(uploaded_file)

    # Convert image to grayscale
    grayscale_image = original_image.convert("L")

    # Convert image to NumPy array
    image_array = np.array(
        grayscale_image,
        dtype=np.float64
    )

    # Create averaging kernel
    kernel = create_average_kernel(
        kernel_size
    )

    # Apply spatial-domain convolution
    spatial_result = spatial_blur(
        image_array,
        kernel
    )

    # Apply Fourier-domain filtering
    fourier_result = fourier_blur(
        image_array,
        kernel
    )

    # Compare both results
    difference, mae, mse, max_difference = compare_results(
        spatial_result,
        fourier_result
    )

    # -----------------------------------------------------
    # Display Main Results
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")

        st.image(
            grayscale_image,
            use_container_width=True
        )

    with col2:
        st.subheader("Spatial Domain")

        st.image(
            spatial_result / 255.0,
            use_container_width=True,
            clamp=True
        )

    with col3:
        st.subheader("Fourier Domain")

        st.image(
            fourier_result / 255.0,
            use_container_width=True,
            clamp=True
        )

    # -----------------------------------------------------
    # Difference Image
    # -----------------------------------------------------

    st.subheader("Difference Between Results")

    # Normalize only for visualization
    if np.max(difference) > 0:
        difference_visual = difference / np.max(difference)
    else:
        difference_visual = difference

    st.image(
        difference_visual,
        use_container_width=True,
        clamp=True
    )

    st.caption(
        "The difference image is normalized only for visualization. "
        "The actual numerical differences are shown below."
    )

    # -----------------------------------------------------
    # Numerical Validation
    # -----------------------------------------------------

    st.subheader("Numerical Validation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:.3e}"
    )

    col2.metric(
        "Mean Squared Error",
        f"{mse:.3e}"
    )

    col3.metric(
        "Maximum Difference",
        f"{max_difference:.3e}"
    )

    # -----------------------------------------------------
    # Blur Kernel
    # -----------------------------------------------------

    st.subheader("Blur Kernel")

    st.write(kernel)

    # -----------------------------------------------------
    # Conclusion
    # -----------------------------------------------------

    st.subheader("Conclusion")

    st.success(
        """
        The spatial-domain convolution and Fourier-domain
        multiplication produce approximately identical results.

        The very small numerical differences are caused by
        floating-point precision during computation.

        Therefore, this experiment validates the Convolution Theorem.

        Spatial Convolution ≈ Fourier Domain Multiplication
        """
    )


# ---------------------------------------------------------
# Offline Example Output
# ---------------------------------------------------------

st.divider()

st.header("Example Output")

st.write(
    """
    The following images show an offline example of the output
    produced by the application using a sample image and a
    5 x 5 averaging filter.
    """
)


example_original = "results/example_original.png"
example_spatial = "results/example_spatial.png"
example_fourier = "results/example_fourier.png"
example_difference = "results/example_difference.png"


# Check that all example output files exist
example_files = [
    example_original,
    example_spatial,
    example_fourier,
    example_difference
]

if all(os.path.exists(file) for file in example_files):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Example Original Image")

        st.image(
            example_original,
            use_container_width=True
        )

    with col2:
        st.subheader("Example Spatial Blur")

        st.image(
            example_spatial,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Example Fourier Blur")

        st.image(
            example_fourier,
            use_container_width=True
        )

    with col4:
        st.subheader("Example Difference Image")

        st.image(
            example_difference,
            use_container_width=True
        )

    st.caption(
        "These PNG files are saved example outputs from the application."
    )

else:
    st.info(
        "Example output images will appear here when the PNG files "
        "are added to the results folder."
    )