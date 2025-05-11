"""
missing_attacks.py
------------------
Apply the *missing* robustness–evaluation attacks and save the resulting
images in a chosen folder.

Attacks implemented
1) Histogram equalisation / contrast stretching
2) Motion blur
3) Brightness shift  (linear intensity bias)
4) Gamma correction  (non-linear luminance change)
5) JPEG-2000 compression (wavelet‐based)

Author: <your-name>
"""

import os
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> None:
    """Create directory *path* if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_image(img, filename, out_dir):
    """Write *img* as 8-bit PNG in *out_dir* with *filename*."""
    ensure_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, filename), img)


def read_image(path: str):
    """Read an image in colour if available, otherwise as gray-scale."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Cannot open {path}')
    return img


# --------------------------------------------------------------------------- #
#  Individual attack implementations
# --------------------------------------------------------------------------- #
def histogram_equalisation(src):
    """Apply global histogram equalisation (Y channel for colour)."""
    if len(src.shape) == 2:                                   # gray
        return cv2.equalizeHist(src)
    # colour – equalise the luminance (Y) component
    yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def motion_blur(src, ksize: int = 15, angle: float = 0.0):
    """Apply directional motion blur of length *ksize* at *angle* degrees."""
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    # rotate kernel to desired angle
    rot_mat = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (ksize, ksize))
    kernel /= np.sum(kernel)
    return cv2.filter2D(src, -1, kernel)


def brightness_shift(src, beta: int = 30):
    """Add or subtract a constant intensity bias (beta can be negative)."""
    return cv2.convertScaleAbs(src, alpha=1.0, beta=beta)


def gamma_correction(src, gamma: float = 1.5):
    """Apply power-law (gamma) intensity transformation."""
    inv_gamma = 1.0 / gamma
    table = (np.linspace(0, 255, 256) / 255.0) ** inv_gamma * 255
    table = table.astype(np.uint8)
    return cv2.LUT(src, table)


def jpeg2000_compression(src, out_path, rate: int = 500):
    """
    Save *src* as JPEG-2000 then read it back.
    The IMWRITE_JPEG2000_COMPRESSION_X1000 parameter is an integer
    between 0–1000; larger → higher quality.
    """
    ok = cv2.imwrite(out_path, src,
                     [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, rate])
    if not ok:
        raise RuntimeError('OpenCV was built without JPEG-2000 support.')
    # decode back to numpy array (so the caller receives an image, not a file)
    return cv2.imread(out_path, cv2.IMREAD_UNCHANGED)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def apply_missing_attacks(watermarked_path,
                          out_dir='attacked_images_missing'):
    """
    Apply the four (five files) missing attacks to *watermarked_path* and
    store the outputs inside *out_dir*.
    """
    img = read_image(watermarked_path)

    # 1. Histogram equalisation / contrast-stretch
    he_img = histogram_equalisation(img)
    save_image(he_img, 'A16_HistEqual.png', out_dir)

    # 2. Motion blur (15-pixel kernel, 0°)
    mb_img = motion_blur(img, ksize=15, angle=0.0)
    save_image(mb_img, 'A17_MotionBlur.png', out_dir)

    # 3. Brightness shift (+30 intensity)
    bright_img = brightness_shift(img, beta=30)
    save_image(bright_img, 'A18_BrightnessShift.png', out_dir)

    # 4. Gamma correction (γ = 1.5)
    gamma_img = gamma_correction(img, gamma=1.5)
    save_image(gamma_img, 'A19_Gamma15.png', out_dir)

    # 5. JPEG-2000 compression / decompression
    jp2_path = os.path.join(out_dir, 'tmp_jp2.jp2')
    jp2_img = jpeg2000_compression(img, jp2_path, rate=500)
    save_image(jp2_img, 'A20_JPEG2000.png', out_dir)
    # remove the intermediate .jp2 file if desired
    if os.path.exists(jp2_path):
        os.remove(jp2_path)

    print(f'All missing attacks stored in: {out_dir}')


# --------------------------------------------------------------------------- #
#  Example usage
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    apply_missing_attacks('Dataset/1.jpg')
