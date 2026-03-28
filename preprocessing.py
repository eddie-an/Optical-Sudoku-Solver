__all__ = ["create_gaussian_kernel", "linear_filter", "median_filter", "create_histogram", "find_otsu_threshold", "perform_global_threshold", "apply_adaptive_threshold", "harris_corners", "order_points", "find_arc_length", "find_area", "is_cell_empty", "approximate_polygon"]

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

def perform_global_threshold(image, threshold):
    """
    Applies global thresholding to a grayscale image.
    Pixels with intensity above the threshold are set to 255, all others to 0.
    Args:
        image (numpy array): Grayscale image.
        threshold (int): Intensity threshold value in range [0, 255].
    Returns:
        numpy array: Binary image of the same shape as input.
    """
    N, M = image.shape
    image_threshold = np.zeros((N, M))
    for i in range(len(image_threshold)):
        for j in range(len(image_threshold[i])):
            if image[i][j] > threshold:
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
