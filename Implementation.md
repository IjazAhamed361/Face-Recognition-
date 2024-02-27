An implementation of the project that compares images using Mean Squared Error (MSE) and Structural Similarity Index (SSIM) in Python. This implementation will include the necessary functions and steps to compare images and display the results.

```python
# Importing required libraries
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2

# Function to calculate Mean Squared Error (MSE) between two images
def mse(imageA, imageB):
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    return error

# Function to compare two images using MSE and SSIM, and display the results
def compare_images(imageA, imageB, title):
    mse_val = mse(imageA, imageB)
    ssim_val = skimage.metrics.structural_similarity(imageA, imageB)
    
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (mse_val, ssim_val))
    
    plot_var = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    
    plot_var = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    
    plt.show()

# Main function to initialize image comparison
def main():
    print("Please keep the required images in the root directory.")
    
    # Taking input for image types (jpg/png)
    input1 = input("Enter the image type (jpg/png): ").lower().strip()
    img_type1 = "jpg"
    img_type2 = "png"

    img_types = [img_type2, img_type1]

    if input1 not in img_types:
        print("Invalid image type. Please enter 'jpg' or 'png'.")
        return
    
    # Taking input for image names (original, reference, and modified)
    img_input = input("Enter the original image's name (without extension): ")
    img_input2 = input("Enter the reference image's name (without extension): ")
    img_input3 = input("Enter the modified image's name (without extension): ")

    # Reading images using OpenCV
    if input1 == img_type1:
        original = cv2.imread(img_input + "." + img_type1)
        ref_image = cv2.imread(img_input2 + "." + img_type1)
        modded = cv2.imread(img_input3 + "." + img_type1)
    elif input1 == img_type2:
        original = cv2.imread(img_input + "." + img_type2)
        ref_image = cv2.imread(img_input2 + "." + img_type2)
        modded = cv2.imread(img_input3 + "." + img_type2)

    # Converting images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    modded = cv2.cvtColor(modded, cv2.COLOR_BGR2GRAY)

    # Displaying original, reference, and modified images
    fig = plt.figure("Images")
    images = ("Original", original), ("Reference", ref_image), ("Modified", modded)

    for (i, (name, image)) in enumerate(images):
        plot_var = fig.add_subplot(1, 3, i + 1)
        plot_var.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")

    plt.show()

    # Comparing images
    compare_images(original, original, "Original vs. Original")
    compare_images(original, ref_image, "Original vs. Reference")
    compare_images(original, modded, "Original vs. Modified")

if __name__ == "__main__":
    main()
```

This implementation includes the following steps:

1. Importing necessary libraries (`skimage`, `matplotlib`, `numpy`, `cv2`).
2. Defining functions for calculating Mean Squared Error (MSE) and comparing images using MSE and Structural Similarity Index (SSIM).
3. Implementing the main function `main()` to initialize image comparison.
4. Taking input for image types (jpg/png) and image names (original, reference, and modified).
5. Reading images using OpenCV and converting them to grayscale.
6. Displaying the original, reference, and modified images.
7. Comparing images using the `compare_images` function and displaying the results.

You can run this Python script after saving it to a `.py` file in the root directory containing the required images. The script will prompt you to enter the image type (jpg/png) and the names of the original, reference, and modified images. It will then display the images and compare them using MSE and SSIM, showing the results in separate plots.
