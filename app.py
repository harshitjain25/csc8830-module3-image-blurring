"""
CSC 8830 - Computer Vision
Module 3 Assignment

Web Application:
Image Blurring using Spatial and Fourier Domain Filtering

Run using:
streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image

from image_filtering import (
    create_average_kernel,
    spatial_blur,
    fourier_blur,
    compare_results
)

st.set_page_config(
    page_title="CSC 8830 - Image Blurring",
    layout="wide"
)

st.title("CSC 8830 - Image Blurring")
st.subheader("Spatial Domain vs Fourier Domain Filtering")

st.write(
    """
    This application demonstrates the Convolution Theorem.

    Image convolution in the spatial domain should produce
    the same result as multiplication in the Fourier domain.
    """
)

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

    original_image = Image.open(uploaded_file)

    grayscale_image = original_image.convert("L")

    image_array = np.array(
        grayscale_image,
        dtype=np.float64
    )

    kernel = create_average_kernel(
        kernel_size
    )

    spatial_result = spatial_blur(
        image_array,
        kernel
    )

    fourier_result = fourier_blur(
        image_array,
        kernel
    )

    difference, mae, mse, max_difference = compare_results(
        spatial_result,
        fourier_result
    )

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

    st.subheader("Difference Between Results")

    st.image(
        difference,
        use_container_width=True,
        clamp=True
    )

    st.subheader("Numerical Validation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:.12f}"
    )

    col2.metric(
        "Mean Squared Error",
        f"{mse:.12f}"
    )

    col3.metric(
        "Maximum Difference",
        f"{max_difference:.12f}"
    )

    st.subheader("Blur Kernel")
    st.write(kernel)

    st.subheader("Conclusion")

    st.success(
        """
        The spatial-domain convolution and Fourier-domain
        multiplication produce approximately identical results.

        Therefore, this experiment validates the Convolution Theorem:

        Spatial Convolution = Fourier Domain Multiplication
        """
    )