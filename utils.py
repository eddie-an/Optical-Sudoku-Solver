import cv2
import numpy as np
from skimage.feature import hog

# HOG parameters — must stay consistent between training and inference
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def normalize_cell(img, canvas_size=40, padding=4):
    """
    Normalizes a digit cell image for HOG feature extraction.

    Handles both Chars74K images (dark digit on white background) and
    Sudoku pipeline cells (white digit on black background) by auto-detecting
    which is the foreground.

    Steps:
      1. Invert if background is light (so digit is always white on black)
      2. Threshold to clean binary
      3. Crop to digit bounding box
      4. Scale to fill inner canvas, re-center with padding

    Args:
        img (np.ndarray): Grayscale image of a single digit cell.
        canvas_size (int): Output image size (canvas_size x canvas_size).
        padding (int): Minimum border around the digit in the output canvas.

    Returns:
        np.ndarray: Normalized binary image of shape (canvas_size, canvas_size).
    """
    # Ensure digit is white (255) on black (0) background
    if img.mean() > 127:
        img = cv2.bitwise_not(img)

    # Threshold to clean binary
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of digit pixels
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = binary[y:y + h, x:x + w]

    # Scale digit to fit within the inner region (canvas minus padding on all sides)
    inner = canvas_size - 2 * padding
    scale = inner / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place digit centered on a blank canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized

    return canvas


def extract_hog_features(imgs):
    """
    Extracts HOG feature vectors from an array of grayscale images.

    Args:
        imgs (np.ndarray): Array of grayscale images, shape (N, H, W).

    Returns:
        np.ndarray: Feature matrix of shape (N, num_features).
    """
    features = []
    for img in imgs:
        img_norm = img.astype(np.float32) / 255.0
        feat = hog(img_norm, **HOG_PARAMS)
        features.append(feat)
    return np.array(features)
