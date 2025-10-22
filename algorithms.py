import cv2
import numpy as np
from matplotlib import pyplot as plt

# === 1. Load Image ===
image = cv2.imread('img.png')
if image is None:
    print("❌ Image not found.")
    exit()

# === 2. Convert to float for processing ===
img = np.float32(image) / 255.0

# === 3. Multi-Scale Retinex (MSR) ===
def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log10(img + 1e-6) - np.log10(blur + 1e-6)

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    return retinex

msr = multi_scale_retinex(img, [15, 80, 250])

# === 4. Normalize and stretch intensity ===
for i in range(3):
    msr[:,:,i] = cv2.normalize(msr[:,:,i], None, 0, 1, cv2.NORM_MINMAX)

# === 5. CLAHE for local contrast enhancement ===
lab = cv2.cvtColor((msr * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
l2 = clahe.apply(l)
merged = cv2.merge((l2, a, b))
enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# === 6. Edge-preserving denoise ===
denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)

# === 7. Unsharp Masking (for final clarity) ===
gauss = cv2.GaussianBlur(denoised, (0, 0), 1.5)
sharp = cv2.addWeighted(denoised, 1.5, gauss, -0.5, 0)

# === 8. Save and display results ===
cv2.imwrite('forensic_enhanced.jpg', (sharp))
print("✅ Forensic enhanced image saved as 'forensic_enhanced.jpg'")

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("After Retinex + CLAHE")
plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Final Forensic Enhanced Image")
plt.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
