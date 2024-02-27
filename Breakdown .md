This Python code compares images using Mean Squared Error (MSE) and Structural Similarity Index (SSIM). It allows you to input the names of the images (original, reference, and modified) and then displays the images along with their MSE and SSIM values.

Here's a breakdown of the code:

1. **Import Libraries**: The code imports necessary libraries including `measure` from `skimage`, `matplotlib.pyplot`, `numpy`, `skimage`, and `cv2`.

2. **Mean Squared Error (MSE) Function**: The `mse` function calculates the MSE between two images.

3. **Compare Images Function**: The `compare_images` function compares two images using MSE and SSIM, and displays them along with their MSE and SSIM values.

4. **Initialization Function**: The `init` function initializes the comparison process by taking input for the image types (jpg/png) and image names (original, reference, and modified).

5. **Main Execution**: The main part of the code initializes the comparison process, converts the images to grayscale, displays the images, and compares them using the `compare_images` function.

6. **Display Results**: The results are displayed showing the original image compared with itself, the original image compared with the reference image, and the original image compared with the modified image.

Note: Ensure that the required images are present in the root directory and are accessible to the script.

If you have any specific questions or need further assistance with this code, feel free to ask!
