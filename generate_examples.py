import os
import cv2
import numpy as np
from PIL import Image

from image_filtering import (
    create_average_kernel,
    spatial_blur,
    fourier_blur,
    compare_results
)


# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)


# Load the sample image
image = cv2.imread(
    "images/sample.jpeg",
    cv2.IMREAD_GRAYSCALE
)

image = image.astype(np.float64)


# Use a 5 x 5 averaging kernel
kernel = create_average_kernel(5)


# Generate spatial result
spatial_result = spatial_blur(
    image,
    kernel
)


# Generate Fourier result
fourier_result = fourier_blur(
    image,
    kernel
)


# Calculate difference
difference, mae, mse, max_difference = compare_results(
    spatial_result,
    fourier_result
)


# Normalize difference only for visualization
if np.max(difference) > 0:
    difference_visual = (
        difference / np.max(difference) * 255
    )
else:
    difference_visual = difference


# Save original
Image.fromarray(
    image.astype(np.uint8)
).save(
    "results/example_original.png"
)


# Save spatial result
Image.fromarray(
    np.clip(
        spatial_result,
        0,
        255
    ).astype(np.uint8)
).save(
    "results/example_spatial.png"
)


# Save Fourier result
Image.fromarray(
    np.clip(
        fourier_result,
        0,
        255
    ).astype(np.uint8)
).save(
    "results/example_fourier.png"
)


# Save normalized difference image
Image.fromarray(
    np.clip(
        difference_visual,
        0,
        255
    ).astype(np.uint8)
).save(
    "results/example_difference.png"
)


print("Example output PNG files created successfully.")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Maximum Difference:", max_difference)