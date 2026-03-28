import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from preprocessing import *

# HOG parameters — must stay consistent between training and inference
# pixels_per_cell=(4,4) gives a 10x10 cell grid on a 40x40 image,
# capturing finer local structure to distinguish similar digits (3/6/8/9)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# LBP parameters — captures loop/curve topology that HOG alone misses
# uniform LBP with P=8, R=1 produces 59 unique pattern bins
LBP_PARAMS = dict(P=8, R=1, method="uniform")
LBP_GRID = 4  # divide image into LBP_GRID x LBP_GRID regions for spatial encoding


def normalize_cell(img, canvas_size=40, padding=4):
    """
    Normalizes a digit cell image for HOG feature extraction.

    Handles both Chars74K images and raw Sudoku pipeline cells — both have
    dark digits on a light background. Auto-detects foreground by checking
    mean pixel value and inverts if necessary.

    Steps:
      1. Invert if background is light (so digit is always white on black)
      2. Threshold to clean binary
      3. Morphological opening to remove small noise blobs
      4. Keep only the largest connected component (removes grid lines/fragments)
      5. Crop to digit bounding box
      6. Scale to fill inner canvas, re-center with padding

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

    # # Morphological opening: erode then dilate to remove small isolated blobs
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # # Keep only the largest connected component (removes grid lines and fragments)
    # num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # if num_labels > 1:
    #     # stats[0] is the background — find the largest foreground component
    #     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #     binary = np.where(labels_map == largest, 255, 0).astype(np.uint8)

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


def augment_image(img, rng=None):
    """
    Applies random augmentations to a normalized cell image to simulate
    scan artifacts seen in real Sudoku puzzle images.

    Augmentations (each applied randomly):
      - Rotation        : ±5° to simulate slight print misalignment
      - Erosion/dilation: randomly thins or thickens ink strokes
      - Gaussian blur   : simulates scan softness
      - Salt-and-pepper : simulates scan grain

    Args:
        img (np.ndarray): Normalized grayscale cell image (output of normalize_cell).
        rng (np.random.Generator, optional): Random number generator for reproducibility.

    Returns:
        np.ndarray: Augmented image of the same shape and dtype.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = img.copy().astype(np.float32)

    # Random rotation ±5°
    angle = rng.uniform(-5, 5)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=0)

    # Random erosion or dilation (ink weight variation)
    if rng.random() < 0.5:
        kernel = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # Gaussian blur
    if rng.random() < 0.5:
        sigma = rng.uniform(0.3, 1.0)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=sigma)

    # Salt-and-pepper noise
    if rng.random() < 0.5:
        noise_level = rng.uniform(0.01, 0.05)
        img[rng.random(img.shape) < noise_level / 2] = 255
        img[rng.random(img.shape) < noise_level / 2] = 0

    return img.astype(np.uint8)


def parse_dat_file(dat_path):
    """
    Parses a Sudoku dataset .dat file and returns the 9x9 grid of digit labels.

    .dat format:
      Line 0: device name
      Line 1: image resolution/format
      Lines 2-10: 9 rows of space-separated digits (0 = empty cell)

    Args:
        dat_path (str): Path to the .dat file.

    Returns:
        list[list[int]]: 9x9 grid where 0 means empty and 1-9 are filled digits.
    """
    with open(dat_path) as f:
        lines = f.readlines()
    grid = []
    for line in lines[2:11]:
        row = [int(x) for x in line.strip().split()]
        grid.append(row)
    return grid


def extract_cells_for_training(image_path, cell_size=40):
    """
    Minimal cell extractor for mining training data from the Sudoku dataset.
    Uses cv2 library functions only — independent of the custom pipeline in
    Image_Processing.ipynb.

    Args:
        image_path (str): Path to a Sudoku puzzle image.
        cell_size (int): Output cell size in pixels (should match CELL_SIZE in training).

    Returns:
        list[np.ndarray] | None: 81 cell images in row-major order (white digit on black
        background), or None if the grid could not be detected.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blurred = linear_filter(img, create_gaussian_kernel(9)) # Custom function with the same parameters
    # blurred = cv2.GaussianBlur(img, (9, 9), 0)

    thresh = apply_adaptive_threshold(blurred, 11, 2, is_inverse=True) # Custom function with the same parameters
    # thresh = cv2.adaptiveThreshold(
    #     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    # )
    
    # Find the largest quadrilateral contour (the Sudoku grid)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    # Custom functions with the same parameters
    for contour in contours[:5]:
        contour = contour[:,0,:]
        perimeter = find_arc_length(contour, is_closed=True)
        epsilon = 0.02 * perimeter
        approx = approximate_polygon(contour, epsilon, is_closed=True)

        if len(approx) == 4:
            grid_contour = approx
            break

    # for cnt in contours[:5]:
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     if len(approx) == 4:
    #         grid_contour = approx
    #         break

    if grid_contour is None:
        return None

    # Order corners: top-left, top-right, bottom-right, bottom-left
    # np.diff gives y-x per point: argmin = top-right, argmax = bottom-left
    pts = grid_contour.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype=np.float32)

    # Perspective warp to a fixed square board
    board_size = 450
    dst = np.array([
        [0, 0], [board_size - 1, 0],
        [board_size - 1, board_size - 1], [0, board_size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))

    # Threshold the warped board (inverse binary: digit = white)
    warped_blur = linear_filter(warped, create_gaussian_kernel(5)) # Custom function with the same parameters
    warped_thresh = apply_adaptive_threshold(warped_blur, 11, 2, is_inverse=True) # Custom function with the same parameters
    
    # warped_blur = cv2.GaussianBlur(warped, (5, 5), 0)
    # warped_thresh = cv2.adaptiveThreshold(
    #     warped_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    # )


    # Slice into 81 cells with 10% margin crop (matches Image_Processing.ipynb)
    cell_px = board_size // 9
    margin = int(cell_px * 0.1)
    cells = []
    for i in range(9):
        for j in range(9):
            cell = warped_thresh[i * cell_px:(i + 1) * cell_px,
                                 j * cell_px:(j + 1) * cell_px]
            cell = cell[margin:-margin, margin:-margin]
            cells.append(cv2.resize(cell, (cell_size, cell_size)))

    return cells  # 81 images, row-major order, white digit on black background


def extract_hog_features(imgs):
    """
    Extracts HOG feature vectors from an array of grayscale images.

    Args:
        imgs (np.ndarray): Array of grayscale images, shape (N, H, W).

    Returns:
        np.ndarray: Feature matrix of shape (N, num_hog_features).
    """
    features = []
    for img in imgs:
        img_norm = img.astype(np.float32) / 255.0
        feat = hog(img_norm, **HOG_PARAMS)
        features.append(feat)
    return np.array(features)


def extract_lbp_features(imgs):
    """
    Extracts spatially-encoded LBP (Local Binary Pattern) feature vectors.

    The image is divided into a LBP_GRID x LBP_GRID grid of regions. A
    normalised histogram of uniform LBP codes is computed per region and
    concatenated, giving spatial awareness alongside texture information.

    LBP captures loop/curve topology (closed vs open contours) that helps
    distinguish visually similar digits such as 3, 6, 8, and 9.

    Args:
        imgs (np.ndarray): Array of grayscale images, shape (N, H, W).

    Returns:
        np.ndarray: Feature matrix of shape (N, LBP_GRID * LBP_GRID * n_bins).
    """
    n_bins = LBP_PARAMS["P"] * (LBP_PARAMS["P"] - 1) + 3  # uniform LBP bin count
    features = []
    for img in imgs:
        lbp = local_binary_pattern(img.astype(np.uint8), **LBP_PARAMS)
        h, w = img.shape
        region_h = h // LBP_GRID
        region_w = w // LBP_GRID
        hist_features = []
        for i in range(LBP_GRID):
            for j in range(LBP_GRID):
                region = lbp[i * region_h:(i + 1) * region_h,
                             j * region_w:(j + 1) * region_w]
                hist, _ = np.histogram(region, bins=n_bins,
                                       range=(0, n_bins), density=True)
                hist_features.append(hist)
        features.append(np.concatenate(hist_features))
    return np.array(features)


def extract_features(imgs):
    """
    Extracts combined HOG + LBP feature vectors for digit classification.

    HOG captures edge/gradient structure (shape of the digit).
    LBP captures local texture topology (open vs closed loops).
    Together they are more discriminative for visually similar digits.

    Feature vector size:
      HOG : 9 * 9 * 2 * 2 * 9  = 2,916  (4x4 pixels/cell, 40x40 image)
      LBP : 4 * 4 * 59          =   944
      Total                     = 3,860

    Args:
        imgs (np.ndarray): Array of grayscale images, shape (N, H, W).

    Returns:
        np.ndarray: Feature matrix of shape (N, 3860).
    """
    return np.concatenate([extract_hog_features(imgs),
                           extract_lbp_features(imgs)], axis=1)
