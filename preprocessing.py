__all__ = ["create_gaussian_kernel", "linear_filter", "median_filter", 
           "create_histogram", "find_otsu_threshold", "perform_global_threshold", 
           "apply_adaptive_threshold", "harris_corners", "order_points", "find_arc_length", 
           "find_area", "is_cell_empty", "approximate_polygon", "extract_sudoku_cells", "rotate_board", 
           "warp_perspective_inverse", "warp_perspective_forward"]

import math
import numpy as np
from numba import njit

def create_gaussian_kernel(L, sigma=None):
    """
    Creates a 2D Gaussian kernel used for image smoothing.
    The kernel values are computed using the Gaussian distribution formula
    and normalized so that all values sum to 1.
    Args
        L (int): Kernel size (must be a positive odd integer).
        sigma (float): Standard deviation of the Gaussian distribution. Defaults to L/6 if  is not provided
    Returns:
        numpy array: Normalized 2D Gaussian kernel of shape (L, L).
    """
    if L % 2 == 0 or L <= 0:
        raise ValueError("Kernel size must be an odd integer and greater than 0")
    if sigma is None:
        sigma = L / 6
    midpoint = L // 2
    kernel = [[0 for _ in range(L)] for _ in range(L)]
    total = 0
    for i in range(L):
        for j in range(L):
            y = (i - midpoint)
            x = (j - midpoint)
            value = (1/(2*math.pi*sigma**2)) * math.exp(-(x**2 + y**2)/(2*sigma**2))
            kernel[i][j] = value
            total += value
    for i in range(L):
        for j in range(L):
            kernel[i][j] /= total # Normalization
    return np.array(kernel)


@njit
def _apply_kernel(padded_image, kernel, padding_size):
    """Helper function for performing kernel operations for linear_filter function."""
    N, M = padded_image.shape
    filtered_image = np.zeros_like(padded_image)

    for i in range(padding_size, N - padding_size):
        for j in range(padding_size, M - padding_size):
            # Explicit slicing for Numba compatibility
            image_window = padded_image[i-padding_size : i+padding_size+1, j-padding_size : j+padding_size+1]
            newPixel = np.sum(image_window * kernel)
            filtered_image[i, j] = newPixel
    return filtered_image


def linear_filter(image, kernel, is_clipped=True):
    """
    Applies a linear filter to an image using convolution with the given kernel.
    Supports both grayscale and RGB/BGR images. For RGB images, the kernel
    is applied independently to each channel.
    Args:
        image (numpy array): Grayscale or RGB/BGR image.
        kernel (numpy array): 2D convolution kernel (must be square with odd dimensions).
        is_clipped (bool): If True, output is clipped to [0, 255] as uint8.
                           Set to False to preserve raw float32 values for gradient operations.
    Returns:
        numpy array: Filtered image of the same shape as input, or None if kernel is invalid.
    """
    def _convolution(image, kernel):
        """Helper function for performing convolution for apply_kernel function."""
        L = len(kernel)
        padding_size = (L - 1) // 2
        padded_image = np.pad(image, pad_width=padding_size, mode='edge').astype(np.float32)
        filtered_image = _apply_kernel(padded_image, kernel, padding_size)
        filtered_image = filtered_image[padding_size: -padding_size, padding_size: -padding_size]
        if is_clipped:
            return np.clip(filtered_image, 0, 255).astype(np.uint8)
        else:
            return filtered_image # Returns raw float32 values
    
    if len(kernel) % 2 == 0 or len(kernel) <= 0:
        raise ValueError("Kernel size must be an odd integer and greater than 0")
    if len(image.shape) == 3: # RGB/BGR images
        channels = [_convolution(image[:, :, i], kernel) for i in range(3)]
        return np.dstack(channels)
    else: # Grayscale images
        return _convolution(image, kernel)


@njit
def _apply_median_kernel(padded_image, L, padding_size):
    """Helper function for performing kernel operations for median_filter function."""
    N, M = padded_image.shape
    im_filtered = np.zeros_like(padded_image)
    for i in range(padding_size, N-padding_size):
        for j in range(padding_size, M-padding_size):
            median = np.median(padded_image[i-padding_size : i+padding_size+1, j-padding_size : j+padding_size+1])
            im_filtered[i, j] = median
    return im_filtered


def median_filter(image, L):
    """
    Applies a median filter to a grayscale image.
    Replaces each pixel with the median value in its surrounding L x L neighborhood.
    Effective at removing salt-and-pepper noise while preserving edges.
    Args:
        image (numpy array): Grayscale image.
        L (int): Kernel size (must be a positive odd integer).
    Returns:
        numpy array: Filtered grayscale image of the same shape, or None if L is invalid.
    """
    if L % 2 == 0 or L <= 0:
        raise ValueError("Kernel size must be an odd integer and greater than 0")
    if len(image.shape) == 2: # only grayscale images
        padding_size = (L-1) // 2
        padded_image = np.pad(image, pad_width=padding_size, mode='edge').astype(np.float32)
        filtered_image = _apply_median_kernel(padded_image, L, padding_size)
        return np.clip(filtered_image[padding_size: -padding_size, padding_size: -padding_size], 0, 255).astype(np.uint8)


def create_histogram(image, bin_size, is_normalized):
    """
    Creates an intensity histogram of a grayscale image.
    Args:
        image (numpy array): Grayscale image.
        bin_size (int): Number of intensity levels (typically 256 for uint8 images).
        is_normalized (bool): If True, histogram values represent frequency ratios instead of counts.
    Returns:
        tuple: (hist, bin) where hist is a list of counts/frequencies and bin is a list of intensity levels.
    """
    N,M = image.shape
    hist = [0 for i in range(bin_size)]
    bin = [i for i in range(bin_size)]
    for i in range(len(image)):
        for j in range(len(image[i])):
            if is_normalized:
                hist[image[i][j]] += (1 / (N*M))
            else:
                hist[image[i][j]] += 1
    return hist, bin


def find_otsu_threshold(hist_norm):
    """
    Finds the optimal global threshold of an image using Otsu's algorithm.
    Iterates over all possible thresholds and selects the one that maximizes
    the between-class variance of the two resulting pixel groups.
    Args:
        hist_norm (list): Normalized intensity histogram (output of create_histogram with is_normalized=True).
    Returns:
        int: Optimal threshold value in range [0, 255].
    """
    def _calculate_between_class_variance(hist_norm, T):
        # Todo: This function needs to be optimized in terms of time complexity
        P1, sum1 = 0, 0
        P2, sum2 = 0, 0
        for i in range(len(hist_norm)):
            if i < T:
                sum1 += (i * hist_norm[i])
                P1 += hist_norm[i]
            else:
                sum2 += (i * hist_norm[i])
                P2 += hist_norm[i]
        if P1 == 0 or P2 == 0:
            return 0
        m1 = sum1 / P1
        m2 = sum2 / P2
        variance = (P1*P2) * (m1-m2)**2
        return variance
    
    max_variance = 0
    best_threshold = 0
    for i in range(len(hist_norm)):
        variance = _calculate_between_class_variance(hist_norm, i) # Todo: This line needs to be optimized
        if variance > max_variance:
            max_variance = variance
            best_threshold = i
    return best_threshold

def perform_global_threshold(image, threshold, is_inverse):
    """
    Applies global thresholding to a grayscale image.
    Pixels with intensity above the threshold are set to 255, all others to 0.
    Args:
        image (numpy array): Grayscale image.
        threshold (int): Intensity threshold value in range [0, 255].
        is_inverse (bool): If True, pixels below the threshold are set to 255 (ink white on black paper).
                        Useful for contour detection downstream.
    Returns:
        numpy array: Binary image of the same shape as input.
    """
    N, M = image.shape
    image_threshold = np.zeros((N, M))
    for i in range(len(image_threshold)):
        for j in range(len(image_threshold[i])):
            if is_inverse:
                if image[i][j] < threshold:
                    image_threshold[i][j] = 255
                else:
                    image_threshold[i][j] = 0
            else:
                if image[i][j] >= threshold:
                    image_threshold[i][j] = 255
                else:
                    image_threshold[i][j] = 0
    return image_threshold

@njit
def _apply_threshold_kernel(padded_image, kernel, padding_size, C, is_inverse):
    """Helper function for performing kernel operations for adaptive thresholding."""
    N, M = padded_image.shape
    threshold_image = np.zeros((N, M), dtype=np.uint8)

    for i in range(padding_size, N - padding_size):
        for j in range(padding_size, M - padding_size):
            # Explicit slicing for Numba compatibility
            image_window = padded_image[i-padding_size : i+padding_size+1, j-padding_size : j+padding_size+1]
            threshold = np.sum(image_window * kernel) - C
            if is_inverse:
                if padded_image[i,j] < threshold:
                    threshold_image[i,j] = 255 # Make ink white (for findContours)
                else:
                    threshold_image[i,j] = 0   # Make paper black
            else:
                if padded_image[i,j] >= threshold:
                    threshold_image[i,j] = 255
                else:
                    threshold_image[i,j] = 0
    return threshold_image


def apply_adaptive_threshold(image, L, C=0, is_inverse=False):
    """
    Applies adaptive Gaussian thresholding to a grayscale image.
    Unlike global thresholding, the threshold is computed locally for each pixel
    using a weighted Gaussian average of its L x L neighborhood, minus a constant C.
    This handles uneven illumination across the image.
    Args:
        image (numpy array): Grayscale image.
        L (int): Kernel size for the local neighborhood (must be a positive odd integer).
        C (float): Constant subtracted from the local mean threshold. Defaults to 0.
        is_inverse (bool): If True, pixels below the threshold are set to 255 (ink white on black paper).
                        Useful for contour detection downstream.
    Returns:
        numpy array: Binary thresholded image of the same shape, or None if L is invalid.
    """
    if L % 2 == 0 or L <= 0:
        raise ValueError("Kernel size must be an odd integer and greater than 0")
    if len(image.shape) != 2: # RGB images give error
        raise ValueError("Image must be in grayscale to apply thresholding")
    kernel = create_gaussian_kernel(L, L/6)
    padding_size = (L - 1) // 2
    padded_image = np.pad(image, pad_width=padding_size, mode='edge').astype(np.float32)
    threshold_image = _apply_threshold_kernel(padded_image, kernel, padding_size, C, is_inverse)
    threshold_image = threshold_image[padding_size: -padding_size, padding_size: -padding_size]
    return threshold_image


@njit
def _compute_harris_response(I_x, I_y, window, window_size, padding_size, k, H, W):
    """ Helper function for Harris Corner algorithm"""
    response = np.zeros((H, W))
    for i in range(padding_size, H-padding_size):
        for j in range(padding_size, W-padding_size):
            I_x_squared = np.square(I_x[i-padding_size: i+padding_size+1, j-padding_size: j+padding_size+1])
            I_y_squared = np.square(I_y[i-padding_size: i+padding_size+1, j-padding_size: j+padding_size+1])
            I_x_y = np.multiply(I_x[i-padding_size: i+padding_size+1, j-padding_size: j+padding_size+1], I_y[i-padding_size: i+padding_size+1, j-padding_size: j+padding_size+1])
            
            for a in range(window_size):
                for b in range(window_size):
                    I_x_squared[a,b] = window[a,b] * I_x_squared[a,b]
                    I_y_squared[a,b] = window[a,b] * I_y_squared[a,b]
                    I_x_y[a,b] = window[a,b] * I_x_y[a,b]

            sum_I_x_squared = np.sum(I_x_squared)
            sum_I_y_squared = np.sum(I_y_squared)
            sum_I_x_I_y = np.sum(I_x_y)
            determinant = sum_I_x_squared * sum_I_y_squared - sum_I_x_I_y * sum_I_x_I_y
            trace = sum_I_x_squared + sum_I_y_squared
            response[i][j] = determinant - (k * trace * trace)
    return response


def harris_corners(img, window_size=3, k=0.04):
    """
    Detects corners in a grayscale image using the Harris Corner Detection algorithm.
    Computes the Harris response at each pixel by applying Sobel gradients and a
    sliding window to estimate local structure. High positive response values indicate corners.
    Args:
        img (numpy array): Grayscale image.
        window_size (int): Size of the sliding window used to accumulate gradient information.
        k (float): Harris sensitivity parameter. Typical values are in range [0.04, 0.06].
    Returns:
        numpy array: Harris response map of the same shape as input (float values).
    """
    H, W = img.shape
    window = np.ones((window_size, window_size))
    padding_size = (window_size-1) // 2

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    I_x = linear_filter(img, kernel_x, False)
    I_y = linear_filter(img, kernel_y, False)

    response = _compute_harris_response(I_x, I_y, window, window_size, padding_size, k, H, W)
    return response


def order_points(pts):
    """
    Orders 4 corner points into a consistent format for perspective transformation.
    The ordering is determined by coordinate sums and differences, making it
    robust to the order in which the points were originally detected.
    Args:
        pts (numpy array): Array of 4 points of shape (4, 2).
    Returns:
        numpy array: Ordered points of shape (4, 2) in the format
                     [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

@njit
def find_arc_length(coordinates, is_closed=False):
    """
    Computes the total arc length of a polyline defined by a sequence of 2D points.
    Sums the Euclidean distances between consecutive points. If the shape is closed,
    the distance from the last point back to the first is also included.
    Args:
        coordinates (numpy array): Array of [x, y] values of shape (N, 2).
        is_closed (bool): If True, includes the closing segment from last point to first.
    Returns:
        float: Total arc length of the polyline.
    """
    distance = 0
    (prev_x, prev_y) = coordinates[0,0], coordinates[0,1]
    for i in range(1, len(coordinates)):
        x = coordinates[i,0]
        y = coordinates[i,1]
        distance += math.sqrt((prev_x-x)**2 + (prev_y-y)**2)
        prev_x = x
        prev_y = y
    if is_closed:
        (x,y) = coordinates[0,0], coordinates[0,1]
        distance += math.sqrt((prev_x-x)**2 + (prev_y-y)**2)
    return distance

@njit
def find_area(coordinates):
    """
    Calculates the area of a polygon using the Shoelace Formula.
    The result may be negative depending on the winding order of the points
    (clockwise vs counter-clockwise). Use abs() if only the magnitude is needed.
    Args:
        coordinates (numpy array): Ordered array of [x, y] vertices of shape (N, 2).
    Returns:
        float: Signed area of the polygon.
    """
    n = len(coordinates)
    area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += (coordinates[i, 0] * coordinates[j,1])
        area -= (coordinates[j, 0] * coordinates[i, 1])
    return area / 2


def approximate_polygon(coordinates, epsilon, is_closed):
    """
    Uses the Ramer–Douglas–Peucker algorithm to reduce the number of coordinate points that represent a curve.
    The algorithm forms a line between the first and last coordinates and finds an intermediate point with the greatest 
    orthogonal distance to the line. 
    If the greatest distance is less than epsilon, all intermediate points are discarded.
    Otherwise, the line is divided into two shorter lines at the intermediate point and the process repeats.
    Args:
        coordinates (numpy array): List of [x,y] values.
        epsilon (float): Threshold for discarding coordinates.
        is_closed (bool): True value indicates that the coordinates form a closed shape.

    Returns:
        numpy array: Reduced coordinate points
    """
    def _recursive_helper(coordinates, epsilon, is_closed):
        """Helper function for performing recursion for approximate_polygon function"""
        maxDistance = 0
        index = 0
        end = len(coordinates)
        
        # coordinates of first and last point
        x1, x2 = coordinates[0][0], coordinates[end-1][0]
        y1, y2 = coordinates[0][1], coordinates[end-1][1]
    
        # Using the equation Ax + By + C = 0
        A = y1 - y2
        B = x2 - x1
        C = -(A*x1 + B*y1)
    
        # Find the point with the maximum distance
        for i in range(1, end-1):
            x_p, y_p = coordinates[i][0], coordinates[i][1]
            if is_closed: # Distance between point 1 and point p
                distance = math.sqrt((x1 - x_p)**2 + (y1 - y_p)**2) # Eucliden distance formula
            else: # Distance between a line and point p
                denom = math.sqrt(A**2 + B**2)
                distance = (abs(A*x_p + B*y_p + C) / denom) if (denom != 0) else 0 # Perpendicular distance formula
            if (distance > maxDistance): # Update maximum distance point and its index
                index = i
                maxDistance = distance
    
        result = np.array([])
        if (maxDistance > epsilon): 
            # If max distance is greater than epsilon, divide the line into two shorter lines
            result1 = _recursive_helper(coordinates[:index+1], epsilon, False)
            result2 = _recursive_helper(coordinates[index:], epsilon, False)
            result = np.concatenate([result1[:len(result1)-1], result2])
        else:
            # Discard all intermediate points since they are ALL less than or equal to epsilon
            result = np.array([[coordinates[0][0], coordinates[0][1]], [coordinates[end-1][0], coordinates[end-1][1]]])
        return result
    
    if epsilon <= 0:
        raise ValueError("Epsilon must be greater than 0")
    if is_closed == True: # Lines form a shape (hence it's closed)
        coordinates = np.append(coordinates, [coordinates[0]], axis=0)
        result = _recursive_helper(coordinates, epsilon, True)
        return result[:-1]
    return _recursive_helper(coordinates, epsilon, False)


def is_cell_empty(cell_image, threshold_percent=0.07):
    """
    Determines whether a Sudoku cell is empty based on ink coverage.
    A 15% margin is cropped from the cell edges before analysis to exclude
    grid lines from the pixel count.
    Args:
        cell_image (numpy array): Inverse binary cell image where ink=255 and paper=0.
        threshold_percent (float): Ink ratio below which the cell is considered empty.
                                   Defaults to 0.07 (7% of pixels are ink).
    Returns:
        bool: True if the cell is empty, False if it contains a digit.
    """
    h, w = cell_image.shape
    margin = int(h * 0.15)
    center_crop = cell_image[margin:-margin, margin:-margin] # cropping the margin off the sides to not count sudoku grid lines
    
    # Count the number of white pixels (ink)
    total_pixels = center_crop.size
    white_pixels = np.count_nonzero(center_crop == 255)
    
    # Calculate the percentage of the crop that is ink
    ink_ratio = white_pixels / total_pixels
    
    # If the ink ratio is smaller than our threshold, it's empty
    return ink_ratio < threshold_percent

    
def extract_sudoku_cells(board):
    """
    Divides a perspective-corrected Sudoku board into 81 individual cells.
    Applies Gaussian blur and adaptive thresholding internally to determine
    which cells are empty. The raw (non-thresholded) cropped cell is returned
    alongside the empty flag for use by the digit classifier downstream.
    Args:
        board (numpy array): Grayscale board image, pre-processed using perspective
                             transform so that only the Sudoku grid is visible.
                             Expected to be a square image divisible by 9 (e.g. 450x450).
    Returns:
        list: 81 tuples of (cell_image, is_empty) in row-major order, where
              cell_image is a cropped grayscale cell with grid lines removed,
              and is_empty is a bool indicating whether the cell contains a digit.
    """
    N, M = board.shape

    # Blur and threshold the warped board to get clean binary cells for empty detection
    blurred_board = linear_filter(board, create_gaussian_kernel(9, 1.6), is_clipped=True)
    thresh = apply_adaptive_threshold(blurred_board, 11, 2, is_inverse=True)

    cells = []
    cell_h = N // 9
    cell_w = M // 9

    for i in range(9):   # Rows
        for j in range(9):  # Columns

            # Compute pixel boundaries for this cell
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w

            # Slice the raw (non-thresholded) cell for the classifier
            cell = board[y_start:y_end, x_start:x_end]

            # Crop 10% margin on all sides to exclude grid lines from the digit region
            margin_y = int(cell_h * 0.1)
            margin_x = int(cell_w * 0.1)
            cell_cropped = cell[margin_y:-margin_y, margin_x:-margin_x]

            # Use the thresholded cell (not the cropped one) for empty detection
            # since is_cell_empty applies its own internal margin crop
            isEmpty = is_cell_empty(thresh[y_start:y_end, x_start:x_end], threshold_percent=0.07)

            cells.append((cell_cropped, isEmpty))

    return cells  # 81 (cell_image, is_empty) tuples in row-major order

    
def rotate_board(board, angle):
    """Rotates a square board image by a multiple of 90 degrees.
    Args:
        board (numpy array): Square grayscale board image.
        angle (int): Rotation angle, must be one of [0, 90, 180, 270].
    Returns:
        numpy array: Rotated board image of the same shape.
    """
    rotations = {0: 0, 90: 1, 180: 2, 270: 3}
    k = rotations.get(angle, 0)
    return np.rot90(board, k)


@njit(cache=True)
def bilinear_interpolate(image, x, y):
    """
    Performs bilinear interpolation to compute the pixel value at non-integer coordinates.
    Uses the four nearest pixel values to calculate a weighted average based on distance.
    (used only in the custom warp perspective functions, not in the main pipeline since OpenCV's built-in functions are more efficient)
    Args:
        image (numpy array): Input image from which to interpolate pixel values.
        x (float): X-coordinate of the target pixel (can be non-integer).
        y (float): Y-coordinate of the target pixel (can be non-integer).
    Returns:
        int: Interpolated pixel value at (x, y) as an integer in range [0, 255].
    """
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))

    if x1 == x2 and y1 == y2:
        return image[y1, x1]
    
    Q11 = image[y1, x1] if 0 <= y1 < image.shape[0] and 0 <= x1 < image.shape[1] else 0
    Q12 = image[y1, x2] if 0 <= y1 < image.shape[0] and 0 <= x2 < image.shape[1] else 0
    Q21 = image[y2, x1] if 0 <= y2 < image.shape[0] and 0 <= x1 < image.shape[1] else 0
    Q22 = image[y2, x2] if 0 <= y2 < image.shape[0] and 0 <= x2 < image.shape[1] else 0

    R1 = ((x2 - x) * Q11 + (x - x1) * Q12) if (x2 - x) + (x - x1) != 0 else 0
    R2 = ((x2 - x) * Q21 + (x - x1) * Q22) if (x2 - x) + (x - x1) != 0 else 0

    P = ((y2 - y) * R1 + (y - y1) * R2) if (y2 - y) + (y - y1) != 0 else 0

    if P < 0:
        return 0
    if P > 255:
        return 255
    return int(P)


@njit(cache=True)
def nearest_neighbor_interpolate(image, x, y):
    """
    Performs nearest neighbor interpolation to compute the pixel value at non-integer coordinates.
    Rounds the coordinates to the nearest integer pixel and returns that pixel's value.
    (used only in the custom warp perspective functions, not in the main pipeline since OpenCV's built-in functions are more efficient)
    Args:
        image (numpy array): Input image from which to interpolate pixel values.
        x (float): X-coordinate of the target pixel (can be non-integer).
        y (float): Y-coordinate of the target pixel (can be non-integer).
    Returns:
        int: Interpolated pixel value at (x, y) as an integer in range [0, 255].
    """
    x_nearest = int(round(x))
    y_nearest = int(round(y))
    
    if 0 <= y_nearest < image.shape[0] and 0 <= x_nearest < image.shape[1]:
        return image[y_nearest, x_nearest]
    else:
        return 0
    
@njit(cache=True)
def int_interpolate(image, x, y):
    """
    Performs integer interpolation to compute the pixel value at non-integer coordinates.
    Rounds the coordinates down to the nearest integer pixel and returns that pixel's value.
    (used only in the custom warp perspective functions, not in the main pipeline since OpenCV's built-in functions are more efficient)
    Args:
        image (numpy array): Input image from which to interpolate pixel values.
        x (float): X-coordinate of the target pixel (can be non-integer).
        y (float): Y-coordinate of the target pixel (can be non-integer).
    Returns:
        int: Interpolated pixel value at (x, y) as an integer in range [0, 255].
    """
    x_int = int(np.floor(x))
    y_int = int(np.floor(y))
    
    if 0 <= y_int < image.shape[0] and 0 <= x_int < image.shape[1]:
        return image[y_int, x_int]
    else:
        return 0
@njit(cache=True)
def warp_perspective_inverse(image, matrix, size, interpolation='bilinear'):
    """
    Applies a perspective transformation to an image given source and destination points.
    This is used to obtain a top-down view of the Sudoku board for consistent cell extraction.
    Uses inverse mapping (backward mapping) with interpolation for smooth results.
    Args:
        image (numpy array): Input image to be warped.
        matrix (numpy array): 3x3 perspective transformation matrix.
        size (tuple): Desired output size (width, height) of the warped image.
        interpolation (str): Interpolation method: 'bilinear' (default), 'nearest', or 'int'.

    Returns:
        numpy array: Warped image of the specified size.

    """
    h, w = size
    warped_image = np.zeros((h, w), dtype=np.uint8)
    # Get the inverse matrix for mapping destination pixels back to source pixels
    inverse = np.linalg.inv(matrix)

    for y in range(h):
        for x in range(w):
            x_prime = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
            y_prime = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
            w_prime = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
            if w_prime != 0:
                x_src = x_prime / w_prime
                y_src = y_prime / w_prime
                if 0 <= x_src < image.shape[1] and 0 <= y_src < image.shape[0]:
                    if interpolation == 'nearest':
                        warped_image[y, x] = nearest_neighbor_interpolate(image, x_src, y_src)
                    elif interpolation == 'int':
                        warped_image[y, x] = int_interpolate(image, x_src, y_src)
                    else:
                        warped_image[y, x] = bilinear_interpolate(image, x_src, y_src)

    return warped_image

@njit(cache=True)
def warp_perspective_forward(image, matrix, size, interpolation='nearest'):
    """
    Applies a perspective transformation to an image given source and destination points.
    Uses forward mapping (source-to-destination) which may have gaps.
    This is used to obtain a top-down view of the Sudoku board for consistent cell extraction.
    Args:
        image (numpy array): Input image to be warped.
        matrix (numpy array): 3x3 perspective transformation matrix.
        size (tuple): Desired output size (width, height) of the warped image.
        interpolation (str): Interpolation method: 'nearest' (default) or 'int'.
    Returns:
        numpy array: Warped image of the specified size.
    """
    h, w = size
    if interpolation not in ('nearest', 'int'):
        raise ValueError("interpolation must be 'nearest' or 'int' for forward transform")

    warped_image = np.zeros((h, w), dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            x_prime = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
            y_prime = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
            w_prime = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
            if w_prime != 0:
                x_norm = x_prime / w_prime
                y_norm = y_prime / w_prime

                if interpolation == 'nearest':
                    x_dst = int(round(x_norm))
                    y_dst = int(round(y_norm))
                else:  # 'int': floor rounding
                    x_dst = int(x_norm)
                    y_dst = int(y_norm)

                if 0 <= x_dst < w and 0 <= y_dst < h:
                    warped_image[y_dst, x_dst] = image[y, x]

    return warped_image