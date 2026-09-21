"""
CSC 8830 - Computer Vision
Module 3 Assignment

Image Blurring using Spatial and Fourier Domain Filtering.

This script implements:
1. Averaging blur kernel creation.
2. Spatial-domain convolution.
3. Fourier-domain convolution.
4. Comparison between both outputs.

The goal is to demonstrate the Convolution Theorem:

Convolution in the spatial domain is equivalent to
multiplication in the frequency domain.
"""

import numpy as np
from scipy.signal import convolve2d


def create_average_kernel(size):
    """
    Create a normalized averaging filter.
    """

    kernel = np.ones((size, size), dtype=np.float64)
    kernel = kernel / kernel.sum()

    return kernel


def spatial_blur(image, kernel):
    """
    Blur an image using direct spatial-domain convolution.
    """

    result = convolve2d(
        image,
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0
    )

    return result


def fourier_blur(image, kernel):
    """
    Blur an image using Fourier-domain multiplication.

    Convolution Theorem:
    f * h = IFFT(FFT(f) * FFT(h))
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height + kernel_height - 1
    output_width = image_width + kernel_width - 1

    image_fft = np.fft.fft2(
        image,
        s=(output_height, output_width)
    )

    kernel_fft = np.fft.fft2(
        kernel,
        s=(output_height, output_width)
    )

    multiplied = image_fft * kernel_fft

    full_result = np.fft.ifft2(multiplied)
    full_result = np.real(full_result)

    start_y = (kernel_height - 1) // 2
    start_x = (kernel_width - 1) // 2

    result = full_result[
        start_y:start_y + image_height,
        start_x:start_x + image_width
    ]

    return result


def compare_results(spatial_result, fourier_result):
    """
    Compare spatial-domain and Fourier-domain results.
    """

    difference = np.abs(
        spatial_result - fourier_result
    )

    mae = np.mean(difference)

    mse = np.mean(
        (spatial_result - fourier_result) ** 2
    )

    max_difference = np.max(difference)

    return difference, mae, mse, max_difference