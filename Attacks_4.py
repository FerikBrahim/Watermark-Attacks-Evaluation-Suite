"""
Attack-suite generator for watermark-robustness testing
Author: <your-name>
Date  : <yyyy-mm-dd>
"""
import os
import math
import cv2
import numpy as np
from skimage import exposure, util
from skimage.filters import gaussian, unsharp_mask, median
from skimage.morphology import square
from PIL import Image, ImageEnhance
import imageio.v2 as imageio

# -----------------------------------------------------------
# 0. I/O settings
# -----------------------------------------------------------
INPUT_IMG  = "watermarked.png"         # <- path to your WM image
OUT_DIR    = "attacks_out"             # <- folder to store outputs
os.makedirs(OUT_DIR, exist_ok=True)

# detect extension for saving
ext = os.path.splitext(INPUT_IMG)[-1].lstrip('.')

# read image as RGB uint8
orig = cv2.cvtColor(cv2.imread(INPUT_IMG, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
h, w  = orig.shape[:2]


# -----------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------
def save(img, tag):
    """Save numpy RGB array with given tag."""
    fname = f"{os.path.splitext(os.path.basename(INPUT_IMG))[0]}__{tag}.{ext}"
    imageio.imwrite(os.path.join(OUT_DIR, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def rand_pad(img, tx, ty):
    """Pad / translate image with empty (black) pixels."""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (w, h), borderValue=(0,0,0))
    return shifted

def clip_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

# -----------------------------------------------------------
# ATTACK A1 – JPEG compression (OpenCV imencode quality param)
# -----------------------------------------------------------
for q in (10, 50, 90):
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    jpeg = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
    save(jpeg, f"A01_JPEG_Q{q}")

# -----------------------------------------------------------
# ATTACK A2 – JPEG-2000 compression
# -----------------------------------------------------------
for rate in (0.1, 0.5, 1.0):                 # bits per pixel ≈
    _, enc = cv2.imencode('.jp2', cv2.cvtColor(orig, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(rate*1000)])
    jp2 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    save(cv2.cvtColor(jp2, cv2.COLOR_BGR2RGB), f"A02_JP2_bpp{rate}")

# -----------------------------------------------------------
# ATTACK A3 – Additive white Gaussian noise
# -----------------------------------------------------------
for s in (5, 15, 30):
    noisy = util.random_noise(orig/255.0, mode='gaussian', var=(s/255.0)**2)
    save((noisy*255).astype(np.uint8), f"A03_AWGN_sigma{s}")

# -----------------------------------------------------------
# ATTACK A4 – Salt & pepper noise
# -----------------------------------------------------------
for d in (0.01, 0.05, 0.10):
    sp = util.random_noise(orig/255.0, mode='s&p', amount=d)
    save((sp*255).astype(np.uint8), f"A04_SaltPepper_{int(d*100)}pct")

# -----------------------------------------------------------
# ATTACK A5 – Median filtering
# -----------------------------------------------------------
for k in (3, 5):
    med = median(orig, square(k))
    save(med, f"A05_Median_{k}x{k}")

# -----------------------------------------------------------
# ATTACK A6 – Gaussian blur
# -----------------------------------------------------------
for sig in (0.5, 1.5, 3.0):
    blur = gaussian(orig, sigma=sig, channel_axis=-1, preserve_range=True)
    save(clip_uint8(blur), f"A06_GBlur_sigma{sig}")

# -----------------------------------------------------------
# ATTACK A7 – Sharpening (un-sharp mask)
# -----------------------------------------------------------
for amt in (0.5, 1.5, 2.0):
    sharp = unsharp_mask(orig, radius=1, amount=amt, channel_axis=-1, preserve_range=True)
    save(clip_uint8(sharp), f"A07_Sharpen_amt{amt}")

# -----------------------------------------------------------
# ATTACK A8 – Histogram equalisation / CLAHE
# -----------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = cv2.merge([clahe.apply(ch) for ch in cv2.split(cv2.cvtColor(orig, cv2.COLOR_RGB2LAB))])
save(cv2.cvtColor(clahe_img, cv2.COLOR_Lab2RGB), "A08_CLAHE")

# -----------------------------------------------------------
# ATTACK A9 – Gamma adjustment
# -----------------------------------------------------------
for g in (0.6, 1.4):
    gamma = np.power(orig/255.0, g)
    save(clip_uint8(gamma*255), f"A09_Gamma_{g}")

# -----------------------------------------------------------
# ATTACK A10 – Scaling (down / up)
# -----------------------------------------------------------
for s in (0.25, 0.5, 2.0):
    resized = cv2.resize(orig, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    scaled = cv2.resize(resized, (w, h), interpolation=cv2.INTER_CUBIC)
    save(scaled, f"A10_Scale_{s}")

# -----------------------------------------------------------
# ATTACK A11 – Rotation
# -----------------------------------------------------------
for ang in (1, 5, 15):
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    rot = cv2.warpAffine(orig, M, (w, h), borderValue=(0,0,0))
    save(rot, f"A11_Rot_{ang}deg")

# -----------------------------------------------------------
# ATTACK A12 – Cropping & repadding
# -----------------------------------------------------------
for pct in (0.05, 0.15):
    dx, dy = int(w*pct), int(h*pct)
    crop = orig[dy:h-dy, dx:w-dx]
    pad = cv2.copyMakeBorder(crop, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(0,0,0))
    save(cv2.resize(pad, (w, h)), f"A12_Crop_{int(pct*100)}pct")

# -----------------------------------------------------------
# ATTACK A13 – Translation
# -----------------------------------------------------------
for shift in (5, 20):
    save(rand_pad(orig, shift, shift), f"A13_Translate_{shift}px")

# -----------------------------------------------------------
# ATTACK A14 – Affine / shear
# -----------------------------------------------------------
for sh in (0.1, 0.2):
    M = np.float32([[1, sh, 0],
                    [0,  1 , 0]])
    shear = cv2.warpAffine(orig, M, (int(w+sh*h), h), borderValue=(0,0,0))
    shear = cv2.resize(shear, (w, h))
    save(shear, f"A14_Shear_{sh}")

# -----------------------------------------------------------
# ATTACK A15 – Colour quantisation (GIF-like)
# -----------------------------------------------------------
for n_col in (64, 16):
    pil = Image.fromarray(orig)
    gif = pil.convert('P', palette=Image.ADAPTIVE, colors=n_col).convert('RGB')
    save(np.array(gif), f"A15_Quant_{n_col}c")

# -----------------------------------------------------------
# ATTACK A16 – Chroma subsampling 4:2:0
# -----------------------------------------------------------
yuv = cv2.cvtColor(orig, cv2.COLOR_RGB2YCrCb)
# downsample chroma
y, cr, cb = cv2.split(yuv)
cr_ds = cv2.resize(cr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
cb_ds = cv2.resize(cb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
# upsample back
cr_up = cv2.resize(cr_ds, (w, h), interpolation=cv2.INTER_LINEAR)
cb_up = cv2.resize(cb_ds, (w, h), interpolation=cv2.INTER_LINEAR)
yuv420 = cv2.merge([y, cr_up, cb_up])
save(cv2.cvtColor(yuv420, cv2.COLOR_YCrCb2RGB), "A16_Chroma420")

# -----------------------------------------------------------
# ATTACK A17 – Row / column removal
# -----------------------------------------------------------
drop_pct = 0.03
rows_to_drop = np.random.choice(h, size=int(h*drop_pct), replace=False)
cols_to_drop = np.random.choice(w, size=int(w*drop_pct), replace=False)
row_mask = np.ones((h,w,3), dtype=bool)
row_mask[rows_to_drop, :, :] = False
col_mask = np.ones((h,w,3), dtype=bool)
col_mask[:, cols_to_drop, :] = False
loss = orig.copy()
loss[~row_mask | ~col_mask] = 0
save(loss, f"A17_PacketLoss_{int(drop_pct*100)}pct")

# -----------------------------------------------------------
# ATTACK A18 – Re-watermarking (overlay another low-power mark)
# -----------------------------------------------------------
# simple text overlay as 2nd "watermark"
overlay = orig.copy()
cv2.putText(overlay, "2ndWM", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2, cv2.LINE_AA)
alpha = 0.15
rewm = cv2.addWeighted(orig, 1-alpha, overlay, alpha, 0)
save(rewm, "A18_Rewatermark")

print(f"Finished. All attacked images stored in: {OUT_DIR}")
