import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: The first image.
        img2: The second image.

    Returns:
        The PSNR value.
    """

    mse = np.mean((img1 - img2) ** 2)
    max_pixel_value = np.max(img1)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.

    Returns:
        The SSIM value.
    """

    ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())
    return ssim_value

def calculate_ncc(img1, img2):
    """Calculates the Normalized Cross-Correlation (NCC) between two images.

    Args:
        img1: The first image.
        img2: The second image.

    Returns:
        The NCC value.
    """

    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    std1 = np.std(img1)
    std2 = np.std(img2)

    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = std1 * std2 * img1.shape[0] * img1.shape[1]

    ncc = numerator / denominator
    return ncc

# Load the images
img1 = cv2.imread('liftingbody.png')
img2 = cv2.imread('watermarked_image.png')

# Ensure images are in the same color space (e.g., grayscale)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate PSNR, SSIM, and NCC
psnr = calculate_psnr(img1, img2)
ssim_value = calculate_ssim(img1, img2)
ncc = calculate_ncc(img1, img2)

print("PSNR:", psnr)
print("SSIM:", ssim_value)
print("NCC:", ncc)