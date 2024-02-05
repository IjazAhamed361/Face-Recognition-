# Face-Recognition-
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2

def mse(imageA, imageB):
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    return error

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

print("Keep The required images in the root directory!!")

def init():
    input1 = input("Enter the Image type:[jpg/png]: ")
    input1 = input1.lower()
    input1 = input1.strip()

    img_type1 = "jpg"
    img_type2 = "png"

    img_types = [img_type2, img_type1]

    if input1 not in img_types:
        print("Enter a Valid image type[jpg/png] ")
        init()
    elif input1 in img_types:
        img_input = input("Enter Original Image's Name: ")
        img_input2 = input("Enter Reference Image's Name: ")
        img_input3 = input("Enter Modified Image's Name: ")

        global original 
        global ref_image
        global modded

        if input1 == img_type1:
            original = cv2.imread(img_input + "." + img_type1)
            ref_image = cv2.imread(img_input2 + "." + img_type1)
            modded = cv2.imread(img_input3 + "." + img_type1)
        elif input1 == img_type2:
            original = cv2.imread(img_input + "." + img_type2)
            ref_image = cv2.imread(img_input2 + "." + img_type2)
            modded = cv2.imread(img_input3 + "." + img_type2)

init()
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
modded = cv2.cvtColor(modded, cv2.COLOR_BGR2GRAY)

fig = plt.figure("Images")
images = ("Original", original), ("Reference", ref_image), ("Modified", modded)

for (i, (name, image)) in enumerate(images):
    plot_var = fig.add_subplot(1, 3, i + 1)
    plot_var.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

plt.show()

compare_images(original, original, "Original vs. Original")
compare_images(original, ref_image, "Original vs. Reference")
compare_images(original, modded, "Original vs. Modified")
