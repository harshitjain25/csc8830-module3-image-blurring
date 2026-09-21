# CSC 8830 - Computer Vision

## Module 3 Assignment

### Image Blurring Using Spatial and Fourier Domain Filtering

This project demonstrates the Convolution Theorem using image blurring.

The same blur operation is performed using:

1. Spatial-domain convolution
2. Fourier-domain multiplication

The outputs are compared visually and numerically.

## Live Web Application

The deployed Streamlit application can be accessed here:

[Open the Streamlit App](https://csc8830-module3-image-blurring-7uxpwmyzhcocvh5fgvnbrs.streamlit.app/)

## Convolution Theorem

Convolution in the spatial domain is equivalent to multiplication in the frequency domain:

```text
f(x,y) * h(x,y) = IFFT(F(u,v)H(u,v))
```

## Features

- Upload an image
- Select blur kernel size
- Spatial-domain image filtering
- Fourier-domain image filtering
- Difference visualization
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Maximum Difference
- Display blur kernel

## Technologies Used

- Python
- NumPy
- SciPy
- Pillow
- OpenCV
- Streamlit

## Project Structure

```text
csc8830-module3-image-blurring/
│
├── app.py
├── image_filtering.py
├── test.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
│   └── sample.jpeg
│
└── results/
```

## How to Run

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

The program applies the same averaging blur filter using two approaches.

### Spatial Domain

The image is directly convolved with the blur kernel:

```text
g(x,y) = f(x,y) * h(x,y)
```

### Fourier Domain

The image and kernel are transformed into the frequency domain. Their Fourier transforms are multiplied, and the inverse Fourier transform is applied:

```text
G(u,v) = F(u,v)H(u,v)

g(x,y) = IFFT(G(u,v))
```

According to the Convolution Theorem:

```text
f(x,y) * h(x,y) = IFFT(F(u,v)H(u,v))
```

## Validation

The spatial-domain result and Fourier-domain result are compared using:

- Mean Absolute Error
- Mean Squared Error
- Maximum Difference
- Difference image

The errors should be extremely close to zero. Small non-zero values may occur because of floating-point numerical precision.

This demonstrates that spatial convolution and Fourier-domain multiplication produce equivalent results.

## Author

Harshit Jain  
CSC 8830 - Computer Vision  
Georgia State University
