# CSC 8830 - Computer Vision
## Module 3 Assignment

### Image Blurring Using Spatial and Fourier Domain Filtering

This project demonstrates image blurring using both spatial-domain convolution and Fourier-domain multiplication.

The purpose of the assignment is to show that convolution in the spatial domain gives the same result as multiplication in the frequency domain, apart from very small floating-point numerical differences.

## Convolution Theorem

The Convolution Theorem states:

```text
f(x,y) * h(x,y) = IFFT(F(u,v)H(u,v))
```

where:

- `f(x,y)` is the input image
- `h(x,y)` is the blur filter
- `*` represents convolution
- `F(u,v)` is the Fourier transform of the image
- `H(u,v)` is the Fourier transform of the filter
- `IFFT` is the inverse Fourier transform

## Features

- Upload a JPG, JPEG, or PNG image
- Convert the uploaded image to grayscale
- Select blur kernel size
- Apply spatial-domain convolution
- Apply Fourier-domain multiplication
- Display both blurred images
- Display a normalized difference image
- Show Mean Absolute Error
- Show Mean Squared Error
- Show Maximum Difference
- Display the averaging blur kernel

## Numerical Validation

The spatial-domain and Fourier-domain results are compared using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Maximum Difference

The values are expected to be extremely close to zero.

Small non-zero values are caused by floating-point precision during numerical computation.

The difference image shown in the web application is normalized only for visualization. The actual numerical differences remain extremely small.

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

## Installation

Clone the repository:

```bash
git clone https://github.com/harshitjain25/csc8830-module3-image-blurring.git
```

Enter the project folder:

```bash
cd csc8830-module3-image-blurring
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## Recommended requirements.txt

For Streamlit Community Cloud deployment, use only the packages required by this project:

```text
streamlit
numpy
scipy
pillow
opencv-python-headless
```

## GitHub Repository

https://github.com/harshitjain25/csc8830-module3-image-blurring

## Web Application

https://csc8830-module3-image-blurring-7uxpwmyzhcocvh5fgvnbrs.streamlit.app/

## Conclusion

The project shows that image blurring using spatial-domain convolution and Fourier-domain multiplication produces approximately identical results.

The very small numerical differences are caused by floating-point precision.

This experimentally validates the Convolution Theorem.

## Author

Harshit Jain  
CSC 8830 - Computer Vision  
Georgia State University
