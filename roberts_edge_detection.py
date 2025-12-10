

import sys
import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(img):
    if img.ndim == 2:
        return img.astype(float)
    r, g, b = img[...,0], img[...,1], img[...,2]
    return (0.2989*r + 0.5870*g + 0.1140*b).astype(float)


def convolve2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    kernel = np.flipud(np.fliplr(kernel))
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    out = np.zeros_like(image, dtype=float)

    for y in range(ih):
        for x in range(iw):
            region = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(region * kernel)

    return out


def roberts_operator(gray):
    Kx = np.array([
        [1, 0],
        [0, -1]
    ], dtype=float)

    Ky = np.array([
        [0, 1],
        [-1, 0]
    ], dtype=float)

    Gx = convolve2d(gray, Kx)
    Gy = convolve2d(gray, Ky)
    magnitude = np.hypot(Gx, Gy)
    return Gx, Gy, magnitude


def normalize(img):
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def main(path):
    img = imageio.imread(path)
    gray = rgb2gray(img)

    Gx, Gy, mag = roberts_operator(gray)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(normalize(mag), cmap="gray"); plt.title("Roberts Magnitude"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(normalize(Gy), cmap="gray"); plt.title("Roberts Gy"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    out_path = os.path.splitext(path)[0] + "_roberts.png"
    imageio.imwrite(out_path, normalize(mag))
    print(f"Hasil Roberts disimpan ke: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python roberts_edge_detection.py <image>")
        sys.exit(1)
    main(sys.argv[1])
