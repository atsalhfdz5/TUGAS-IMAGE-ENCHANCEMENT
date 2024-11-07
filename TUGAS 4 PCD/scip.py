import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def histogram_equalization(image):
    """
    Fungsi untuk melakukan ekualisasi histogram pada citra grayscale.
    """
    # Menghitung histogram citra
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # Menghitung CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # Normalisasi CDF untuk memastikan rentang nilai antara 0 dan 255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Gunakan CDF untuk melakukan transformasi pada citra
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    # Bentuk citra yang telah diekualisasi
    image_equalized = image_equalized.reshape(image.shape)
    return image_equalized

def apply_gaussian_blur(image, sigma=1):
    """
    Fungsi untuk menerapkan filter Gaussian ke citra.
    """
    return gaussian_filter(image, sigma=sigma)

# Membaca citra grayscale
path = "C://Users//ACER//Downloads//Burung Hantu.jpeg"
image = imageio.imread(path)

# Mengubah rentang citra menjadi 0-255 (karena as_gray menghasilkan nilai antara 0-1)
image = np.uint8(image * 255)

# Menampilkan citra asli
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Citra Asli")

# Ekualisasi histogram citra
image_eq = histogram_equalization(image)

# Menampilkan citra hasil ekualisasi
plt.subplot(1, 2, 2)
plt.imshow(image_eq, cmap='gray')
plt.title("Citra Setelah Ekualisasi Histogram")
plt.show()

# Menyimpan citra hasil ekualisasi
imageio.imwrite("C://Users//ACER//Downloads//Burung_Hantu_output.jpeg", np.uint8(image_eq))

# Menambahkan efek blur pada citra hasil ekualisasi
image_blurred = apply_gaussian_blur(image_eq, sigma=2)

# Menampilkan citra hasil blur
plt.figure(figsize=(6, 6))
plt.imshow(image_blurred, cmap='gray')
plt.title("Citra Setelah Gaussian Blur")
plt.show()

# Menyimpan citra yang telah diblur
imageio.imwrite("C://Users//ACER//Downloads//Burung_Hantu_blurred.jpeg", np.uint8(image_blurred))
