import numpy as np
import pywt
from PIL import Image

def insert_watermark(image_path, watermark, output_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)

    # Perform 2-level DWT
    coeffs = pywt.wavedec2(img_array, 'haar', level=2)
    LL2, (HL2, LH2, HH2), (HL1, LH1, HH1) = coeffs

    # Convert LL2 to integers
    LL2_int = np.round(LL2).astype(np.int32)

    # Flatten the LL2 subband
    LL2_flat = LL2_int.flatten()

    # Insert watermark using LSB
    for i, bit in enumerate(watermark):
        LL2_flat[i] = (LL2_flat[i] & 0xFE) | (bit & 1)

    # Reshape LL2 back to its original shape
    LL2_watermarked = LL2_flat.reshape(LL2.shape)

    # Convert LL2 back to float
    LL2_watermarked = LL2_watermarked.astype(np.float64)

    # Reconstruct the image
    coeffs_watermarked = (LL2_watermarked, (HL2, LH2, HH2), (HL1, LH1, HH1))
    img_watermarked = pywt.waverec2(coeffs_watermarked, 'haar')

    # Normalize and convert back to uint8
    img_watermarked = np.clip(img_watermarked, 0, 255)
    img_watermarked = img_watermarked.astype(np.uint8)

    # Save the watermarked image
    Image.fromarray(img_watermarked).save(output_path)

def extract_watermark(watermarked_image_path, watermark_size):
    # Load the watermarked image
    img = Image.open(watermarked_image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)

    # Perform 2-level DWT
    coeffs = pywt.wavedec2(img_array, 'haar', level=2)
    LL2, (HL2, LH2, HH2), (HL1, LH1, HH1) = coeffs

    # Convert LL2 to integers
    LL2_int = np.round(LL2).astype(np.int32)

    # Flatten the LL2 subband
    LL2_flat = LL2_int.flatten()

    # Extract watermark using LSB
    extracted_watermark = np.zeros(watermark_size, dtype=np.uint8)
    for i in range(watermark_size):
        extracted_watermark[i] = LL2_flat[i] & 1

    return extracted_watermark

# Example usage
image_path = "Cyst_512.bmp"
output_path = "watermarked_image.bmp"
watermark = np.random.randint(0, 256, 64, dtype=np.uint8)
print(watermark)
insert_watermark(image_path, watermark, output_path)
print("Watermark inserted and image saved successfully!")





# Example usage
image_path = "Cyst_512.bmp"
watermarked_image_path = "watermarked_image.bmp"

watermark_size = 64


# Extract watermark
extracted_watermark = extract_watermark(watermarked_image_path, watermark_size)
print("Watermark extracted successfully!")

# Compare original and extracted watermarks
print("Original watermark:", watermark)
print("Extracted watermark:", extracted_watermark)
print("Watermarks match:", np.array_equal(watermark, extracted_watermark))