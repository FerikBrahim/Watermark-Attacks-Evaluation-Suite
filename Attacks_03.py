# ------------------------------------------------------------------
# 18-Attack Suite for Watermark-Robustness Evaluation
# ------------------------------------------------------------------
#  Requirements
#     pip install opencv-python scikit-image pillow numpy imageio
# ------------------------------------------------------------------
import os, cv2, math, numpy as np
from skimage import util, exposure, filters
from skimage.filters import gaussian, median, unsharp_mask
from skimage.morphology import square
from PIL import Image, ImageEnhance, ImageFilter
import imageio.v2 as imageio

# ------------------------------------------------------------------
#  0. I / O  ----------------------------------------------------------------
# ------------------------------------------------------------------
INPUT_IMG = "Dataset/X-Ray_synpic100312.jpg"        # ← your WM image
OUT_DIR   = "attacks_out_02"                           # ← outputs here
os.makedirs(OUT_DIR, exist_ok=True)

root, ext = os.path.splitext(os.path.basename(INPUT_IMG))
ext = ext.lstrip('.')

# Load as RGB (keeps colour attacks valid; works for gray too)
src = cv2.cvtColor(cv2.imread(INPUT_IMG, cv2.IMREAD_UNCHANGED),
                   cv2.COLOR_BGR2RGB)
H, W = src.shape[:2]

def save(img, tag):
    """Helper: save RGB numpy array with a tag."""
    fname = f"{root}__{tag}.{ext}"
    imageio.imwrite(os.path.join(OUT_DIR, fname),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def clip8(x): return np.clip(x, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------
#  1‒18  ATTACK IMPLEMENTATIONS
# ------------------------------------------------------------------

# A01 – JPEG compression -----------------------------------------------------
for q in (10, 50, 90):
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(src, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, q])
    save(cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB),
         f"A01_JPEG_Q{q}")

# A02 – JPEG-2000 compression -----------------------------------------------
for bpp in (0.1, 0.5, 1.0):
    _, buf = cv2.imencode('.jp2', cv2.cvtColor(src, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
                           int(bpp*1000)])
    save(cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB),
         f"A02_JP2_bpp{bpp}")

# A03 – Additive white Gaussian noise ---------------------------------------
for sigma in (5, 15, 30):
    n = util.random_noise(src/255., mode='gaussian', var=(sigma/255.)**2)
    save((n*255).astype(np.uint8), f"A03_AWGN_s{sigma}")

# A04 – Salt & pepper noise --------------------------------------------------
for amt in (0.01, 0.05, 0.10):
    n = util.random_noise(src/255., mode='s&p', amount=amt)
    save((n*255).astype(np.uint8), f"A04_SP_{int(amt*100)}pct")

# A05 – Median filtering -----------------------------------------------------
for k in (3, 5):
    save(median(src, square(k)), f"A05_Median_{k}")

# A06 – Gaussian blur --------------------------------------------------------
for sig in (0.5, 1.5, 3.0):
    save(clip8(gaussian(src, sigma=sig,
                        channel_axis=-1, preserve_range=True)),
         f"A06_GBlur_{sig}")

# A07 – Sharpening (unsharp mask) -------------------------------------------
for amt in (0.5, 1.5, 2.0):
    sh = unsharp_mask(src, radius=1, amount=amt,
                      channel_axis=-1, preserve_range=True)
    save(clip8(sh), f"A07_Sharp_{amt}")

# A08 – Histogram equalisation / CLAHE --------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab   = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
l,a,b = cv2.split(lab)
lab   = cv2.merge([clahe.apply(l), a, b])
save(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "A08_CLAHE")

# A09 – Gamma correction -----------------------------------------------------
for g in (0.6, 1.4):
    save(clip8(np.power(src/255., g)*255), f"A09_Gamma_{g}")

# A10 – Scaling (down / up) --------------------------------------------------
for s in (0.25, 0.5, 2.0):
    tmp = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    tmp = cv2.resize(tmp, (W, H), interpolation=cv2.INTER_CUBIC)
    save(tmp, f"A10_Scale_{s}")

# A11 – Rotation -------------------------------------------------------------
for ang in (1, 5, 15):
    M = cv2.getRotationMatrix2D((W/2, H/2), ang, 1.0)
    save(cv2.warpAffine(src, M, (W, H),
                        borderValue=(0,0,0)), f"A11_Rot_{ang}")

# A12 – Cropping & padding ---------------------------------------------------
for pct in (0.05, 0.15):
    dy, dx = int(H*pct), int(W*pct)
    crop   = src[dy:H-dy, dx:W-dx]
    pad    = cv2.copyMakeBorder(crop, dy, dy, dx, dx,
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    save(cv2.resize(pad, (W, H)), f"A12_Crop_{int(pct*100)}")

# A13 – Translation ----------------------------------------------------------
for t in (5, 20):
    M = np.float32([[1,0,t], [0,1,t]])
    save(cv2.warpAffine(src, M, (W, H),
                        borderValue=(0,0,0)), f"A13_Trans_{t}")

# A14 – Shear (affine) -------------------------------------------------------
for sh in (0.1, 0.2):
    M = np.float32([[1, sh, 0],
                    [0,  1, 0]])
    out = cv2.warpAffine(src, M, (int(W+sh*H), H),
                         borderValue=(0,0,0))
    out = cv2.resize(out, (W, H))
    save(out, f"A14_Shear_{sh}")

# A15 – Colour quantisation (GIF-like) --------------------------------------
for nc in (64, 16):
    im = Image.fromarray(src)
    q  = im.convert('P', palette=Image.ADAPTIVE, colors=nc).convert('RGB')
    save(np.array(q), f"A15_Quant_{nc}")

# A16 – Chroma subsampling 4:2:0 --------------------------------------------
yuv       = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(yuv)
cr_ds     = cv2.resize(cr, (W//2, H//2), interpolation=cv2.INTER_LINEAR)
cb_ds     = cv2.resize(cb, (W//2, H//2), interpolation=cv2.INTER_LINEAR)
cr_up     = cv2.resize(cr_ds, (W, H),    interpolation=cv2.INTER_LINEAR)
cb_up     = cv2.resize(cb_ds, (W, H),    interpolation=cv2.INTER_LINEAR)
save(cv2.cvtColor(cv2.merge([y, cr_up, cb_up]),
                  cv2.COLOR_YCrCb2RGB), "A16_Chroma420")

# A17 – Packet-loss (drop random rows / cols) -------------------------------
loss = src.copy()
rows = np.random.choice(H, size=int(0.03*H), replace=False)
cols = np.random.choice(W, size=int(0.03*W), replace=False)
loss[rows,:,:] = 0
loss[:,cols,:] = 0
save(loss, "A17_PacketLoss3")

# A18 – Re-watermarking (low-power overlay) ---------------------------------
overlay = src.copy()
cv2.putText(overlay, "2ndWM", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2, cv2.LINE_AA)
alpha = 0.15
save(cv2.addWeighted(src, 1-alpha, overlay, alpha, 0), "A18_ReWM")

print(f"Finished: {len(os.listdir(OUT_DIR))} attacked images saved in '{OUT_DIR}'.")
