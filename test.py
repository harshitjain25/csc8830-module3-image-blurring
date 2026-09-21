import cv2

from image_filtering import (
    create_average_kernel,
    spatial_blur,
    fourier_blur,
    compare_results
)

# Load the sample image in grayscale
image = cv2.imread(
    "images/sample.jpeg",
    cv2.IMREAD_GRAYSCALE
)

# Convert image to floating point
image = image.astype(float)

# Create a 5x5 averaging kernel
kernel = create_average_kernel(5)

# Apply spatial-domain blur
spatial = spatial_blur(image, kernel)

# Apply Fourier-domain blur
fourier = fourier_blur(image, kernel)

# Compare the results
difference, mae, mse, max_diff = compare_results(
    spatial,
    fourier
)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Maximum Difference:", max_diff)