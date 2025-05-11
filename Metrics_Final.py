import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_combined_metrics(L_original, R_original, L_extracted, R_extracted):
    # Ensure all inputs are numpy arrays
    L_original = np.array(L_original)
    R_original = np.array(R_original)
    L_extracted = np.array(L_extracted)
    R_extracted = np.array(R_extracted)

    # Ensure all arrays have the same length
    min_length = min(len(L_original), len(R_original), len(L_extracted), len(R_extracted))
    L_original = L_original[:min_length]
    R_original = R_original[:min_length]
    L_extracted = L_extracted[:min_length]
    R_extracted = R_extracted[:min_length]

    # Combine L and R vectors
    original = np.concatenate((L_original, R_original))
    extracted = np.concatenate((L_extracted, R_extracted))
    
    # Ensure both arrays have the same data type
    original = original.astype(np.float64)
    extracted = extracted.astype(np.float64)
    
    # Bit Error Rate (BER)
    ber = np.mean(original != extracted)
    
    # Mean Squared Error (MSE)
    mse = np.mean((original - extracted) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Structural Similarity Index (SSIM)
    ssim_value = ssim(original.reshape(1, -1), extracted.reshape(1, -1), data_range=255)
    
    # Correlation Coefficient
    corr_coef = np.corrcoef(original, extracted)[0, 1]
    
    # Normalized Cross-Correlation (NCC)
    ncc = np.sum(original * extracted) / (np.sqrt(np.sum(original**2)) * np.sqrt(np.sum(extracted**2)))
    
    return {
        'BER': ber,
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Correlation Coefficient': corr_coef,
        'NCC': ncc
    }

# Example usage:
L_original = np.array([13, 52, 34, 13, 43, 59, 63, 33])
R_original = np.array([36, 3, 2, 22, 27, 0, 16, 59])
L_extracted = np.unpackbits(np.array([13, 52, 34, 13, 43, 59, 63, 33], dtype=np.uint8))
R_extracted = np.unpackbits(np.array([36, 3, 2, 22, 27, 0, 16, 59], dtype=np.uint8))

combined_metrics = calculate_combined_metrics(L_original, R_original, L_extracted, R_extracted)

print("Combined Metrics for Watermark:")
for metric, value in combined_metrics.items():
    print(f"{metric}: {value}")