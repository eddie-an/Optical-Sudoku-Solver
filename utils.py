import json
import os

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

    # Morphological opening: erode then dilate to remove small isolated blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component (removes grid lines and fragments)
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # stats[0] is the background — find the largest foreground component
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels_map == largest, 255, 0).astype(np.uint8)

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


def _extract_cells_for_training_debug(image_path, cell_size=40):
    """
    Runs the training-time Sudoku cell extraction pipeline and returns
    intermediate images for debugging.

    Args:
        image_path (str): Path to a Sudoku puzzle image.
        cell_size (int): Output cell size in pixels.

    Returns:
        dict: Extraction artifacts and metadata. If grid detection fails,
              ``cells`` is None and ``grid_contour`` is None.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "image_path": image_path,
            "original": None,
            "blurred": None,
            "thresh": None,
            "grid_contour": None,
            "contour_overlay": None,
            "ordered_corners": None,
            "warped": None,
            "warped_blur": None,
            "warped_thresh": None,
            "cells": None,
            "cell_debug": [],
            "board_size": 450,
            "cell_px": None,
            "margin": None,
            "error": f"Could not read image: {image_path}",
        }

    blurred = linear_filter(img, create_gaussian_kernel(9))
    thresh = apply_adaptive_threshold(blurred, 11, 2, is_inverse=True)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    for contour in contours[:5]:
        contour = contour[:, 0, :]
        perimeter = find_arc_length(contour, is_closed=True)
        epsilon = 0.02 * perimeter
        approx = approximate_polygon(contour, epsilon, is_closed=True)

        if len(approx) == 4:
            grid_contour = approx
            break

    contour_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if grid_contour is None:
        return {
            "image_path": image_path,
            "original": img,
            "blurred": blurred,
            "thresh": thresh,
            "grid_contour": None,
            "contour_overlay": contour_overlay,
            "ordered_corners": None,
            "warped": None,
            "warped_blur": None,
            "warped_thresh": None,
            "cells": None,
            "cell_debug": [],
            "board_size": 450,
            "cell_px": None,
            "margin": None,
            "error": "Could not find a 4-corner Sudoku grid contour.",
        }

    cv2.drawContours(contour_overlay, [grid_contour.astype(np.int32)], -1, (0, 255, 0), 3)

    pts = grid_contour.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)

    board_size = 450
    dst = np.array([
        [0, 0], [board_size - 1, 0],
        [board_size - 1, board_size - 1], [0, board_size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))

    warped_blur = linear_filter(warped, create_gaussian_kernel(5))
    warped_thresh = apply_adaptive_threshold(warped_blur, 11, 2, is_inverse=True)

    cell_px = board_size // 9
    margin = int(cell_px * 0.1)
    cells = []
    cell_debug = []
    empty_cell_count = 0
    for row in range(9):
        for col in range(9):
            raw_cell = warped_thresh[row * cell_px:(row + 1) * cell_px,
                                     col * cell_px:(col + 1) * cell_px]
            cropped_cell = raw_cell[margin:-margin, margin:-margin]
            resized_cell = cv2.resize(cropped_cell, (cell_size, cell_size))
            normalized_cell = normalize_cell(resized_cell)
            ink_ratio = float(np.count_nonzero(resized_cell == 255) / resized_cell.size)
            is_empty = is_cell_empty(raw_cell, threshold_percent=0.07)
            normalization_failed = normalized_cell.max() == 0

            if is_empty:
                empty_cell_count += 1

            cells.append(resized_cell)
            cell_debug.append({
                "row": row,
                "col": col,
                "raw_cell": raw_cell,
                "cropped_cell": cropped_cell,
                "resized_cell": resized_cell,
                "normalized_cell": normalized_cell,
                "ink_ratio": ink_ratio,
                "is_empty": is_empty,
                "normalization_failed": normalization_failed,
            })

    if empty_cell_count == 81:
        return {
            "image_path": image_path,
            "original": img,
            "blurred": blurred,
            "thresh": thresh,
            "grid_contour": grid_contour,
            "contour_overlay": contour_overlay,
            "ordered_corners": ordered,
            "warped": warped,
            "warped_blur": warped_blur,
            "warped_thresh": warped_thresh,
            "cells": None,
            "cell_debug": cell_debug,
            "board_size": board_size,
            "cell_px": cell_px,
            "margin": margin,
            "error": "Grid contour was found, but all 81 extracted cells were classified as empty.",
        }

    return {
        "image_path": image_path,
        "original": img,
        "blurred": blurred,
        "thresh": thresh,
        "grid_contour": grid_contour,
        "contour_overlay": contour_overlay,
        "ordered_corners": ordered,
        "warped": warped,
        "warped_blur": warped_blur,
        "warped_thresh": warped_thresh,
        "cells": cells,
        "cell_debug": cell_debug,
        "board_size": board_size,
        "cell_px": cell_px,
        "margin": margin,
        "error": None,
    }


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
    debug_data = _extract_cells_for_training_debug(image_path, cell_size=cell_size)
    return debug_data["cells"]


def show_training_cell_extraction_debug(image_path, cell_size=40, suspect_ink_ratio=0.02):
    """
    Displays the original Sudoku image alongside extraction intermediates and
    the 81 extracted cells used by the training pipeline.

    Cells are highlighted when normalization produced a blank result or when
    the extracted cell contains very little ink, which helps explain why some
    samples are being dropped.

    Args:
        image_path (str): Path to the Sudoku puzzle image to inspect.
        cell_size (int): Output cell size used by the extractor.
        suspect_ink_ratio (float): Cells below this ink ratio are marked as
                                   suspicious.

    Returns:
        dict: Debug metadata and intermediate images from the extractor.
    """
    import matplotlib.pyplot as plt

    debug_data = _extract_cells_for_training_debug(image_path, cell_size=cell_size)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.6])

    overview_items = [
        ("Original", debug_data["original"]),
        ("Grid contour", debug_data["contour_overlay"]),
        ("Initial threshold", debug_data["thresh"]),
        ("Warped board", debug_data["warped"]),
    ]

    for idx, (title, image) in enumerate(overview_items):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_title(title)
        if image is None:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=12)
        elif image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    cell_grid = gs[2, :].subgridspec(9, 9, wspace=0.05, hspace=0.25)
    suspicious = []
    for idx in range(81):
        cell_info = debug_data["cell_debug"][idx] if debug_data["cell_debug"] else None
        ax = fig.add_subplot(cell_grid[idx // 9, idx % 9])

        if cell_info is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue

        ax.imshow(cell_info["resized_cell"], cmap="gray", vmin=0, vmax=255)
        title = f"{cell_info['row'] + 1},{cell_info['col'] + 1}"
        is_suspicious = (
            cell_info["normalization_failed"] or
            cell_info["ink_ratio"] < suspect_ink_ratio
        )
        if is_suspicious:
            title += " !"
            suspicious.append(cell_info)
            for spine in ax.spines.values():
                spine.set_color("crimson")
                spine.set_linewidth(2)

        ax.set_title(title, fontsize=8, pad=2)
        ax.axis("off")

    if debug_data["error"] is not None:
        fig.suptitle(
            f"{image_path}\n{debug_data['error']}",
            fontsize=14,
            y=0.98,
        )
    else:
        fig.suptitle(
            f"{image_path}\nSuspicious cells: {len(suspicious)} / 81",
            fontsize=14,
            y=0.98,
        )

    plt.tight_layout()
    plt.show()

    if suspicious:
        print("Suspicious cells")
        for cell_info in suspicious:
            reasons = []
            if cell_info["normalization_failed"]:
                reasons.append("normalize_cell returned blank")
            if cell_info["ink_ratio"] < suspect_ink_ratio:
                reasons.append(f"low ink ratio ({cell_info['ink_ratio']:.3f})")
            print(
                f"  r{cell_info['row'] + 1} c{cell_info['col'] + 1}: "
                + ", ".join(reasons)
            )
    elif debug_data["error"] is None:
        print("No suspicious cells detected by the heuristic.")

    return debug_data


class TrainingImageDebugBrowser:
    """
    Lightweight notebook-friendly browser for stepping through Sudoku dataset
    images, reusing the training-time extraction debug view and persisting
    which images have already been inspected.
    """

    def __init__(
        self,
        dataset_dir,
        history_path=None,
        cell_size=40,
        suspect_ink_ratio=0.02,
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.dataset_dir = dataset_dir
        self.history_path = history_path or os.path.join(
            dataset_dir, ".viewed_debug_images.json"
        )
        self.cell_size = cell_size
        self.suspect_ink_ratio = suspect_ink_ratio
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.image_names = sorted([
            name for name in os.listdir(dataset_dir)
            if name.lower().endswith(self.image_extensions)
        ])
        if not self.image_names:
            raise ValueError(f"No images found in dataset directory: {dataset_dir}")

        self.current_index = 0
        self.viewed = self._load_history()

    def _load_history(self):
        if not os.path.exists(self.history_path):
            return set()
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return set(payload.get("viewed", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _save_history(self):
        payload = {
            "dataset_dir": self.dataset_dir,
            "viewed": sorted(self.viewed),
        }
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _resolve_index(self, index=None, image_name=None):
        if image_name is not None:
            if image_name not in self.image_names:
                raise ValueError(f"Image not found in dataset: {image_name}")
            return self.image_names.index(image_name)
        if index is None:
            return self.current_index
        if not 0 <= index < len(self.image_names):
            raise IndexError(f"Index {index} is out of range for {len(self.image_names)} images.")
        return index

    def _print_status(self):
        current_name = self.image_names[self.current_index]
        viewed_count = len(self.viewed)
        total = len(self.image_names)
        unseen_indices = [
            idx for idx, name in enumerate(self.image_names)
            if name not in self.viewed
        ]
        next_unseen = unseen_indices[0] if unseen_indices else None

        print(
            f"Current: [{self.current_index}] {current_name}\n"
            f"Viewed: {viewed_count}/{total}\n"
            f"History: {self.history_path}"
        )
        if next_unseen is not None:
            print(f"Next unseen: [{next_unseen}] {self.image_names[next_unseen]}")
        else:
            print("Next unseen: none")

    def list_images(self, start=None, limit=20, only_unseen=False):
        """
        Prints a compact window of dataset images with seen/unseen markers.
        """
        if start is None:
            start = max(0, self.current_index - limit // 2)

        end = min(len(self.image_names), start + limit)
        for idx in range(start, end):
            name = self.image_names[idx]
            if only_unseen and name in self.viewed:
                continue
            marker = "x" if name in self.viewed else " "
            current = ">" if idx == self.current_index else " "
            print(f"{current} [{idx:04d}] [{marker}] {name}")

    def show(self, index=None, image_name=None, mark_viewed=True):
        """
        Shows the selected image with extraction debug output.
        """
        resolved_index = self._resolve_index(index=index, image_name=image_name)
        self.current_index = resolved_index
        image_name = self.image_names[resolved_index]
        image_path = os.path.join(self.dataset_dir, image_name)

        debug_data = show_training_cell_extraction_debug(
            image_path,
            cell_size=self.cell_size,
            suspect_ink_ratio=self.suspect_ink_ratio,
        )

        if mark_viewed:
            self.viewed.add(image_name)
            self._save_history()

        self._print_status()
        return debug_data

    def next(self, step=1):
        return self.show(index=min(self.current_index + step, len(self.image_names) - 1))

    def prev(self, step=1):
        return self.show(index=max(self.current_index - step, 0))

    def next_unseen(self):
        for idx in range(self.current_index + 1, len(self.image_names)):
            if self.image_names[idx] not in self.viewed:
                return self.show(index=idx)
        for idx in range(0, self.current_index + 1):
            if self.image_names[idx] not in self.viewed:
                return self.show(index=idx)
        print("All dataset images have been viewed.")
        return None

    def random_unseen(self, rng=None):
        unseen = [name for name in self.image_names if name not in self.viewed]
        if not unseen:
            print("All dataset images have been viewed.")
            return None
        if rng is None:
            rng = np.random.default_rng()
        image_name = unseen[int(rng.integers(0, len(unseen)))]
        return self.show(image_name=image_name)

    def mark_unviewed(self, index=None, image_name=None):
        resolved_index = self._resolve_index(index=index, image_name=image_name)
        image_name = self.image_names[resolved_index]
        self.viewed.discard(image_name)
        self._save_history()
        print(f"Marked unviewed: [{resolved_index}] {image_name}")

    def reset_history(self):
        self.viewed.clear()
        self._save_history()
        print(f"Cleared viewed history: {self.history_path}")


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
