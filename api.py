from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from skimage.measure import approximate_polygon
import joblib
from utils import normalize_cell, extract_features
from preprocessing import *
from solver import solve_sudoku, is_valid_sudoku

# Initialize API
app = FastAPI(title="Optical Sudoku Solver API")

# Define allowed origins
origins = [
    "http://localhost:3000",   # React default
    "http://localhost:5173",   # Vite default
    "http://10.0.0.248:8000",
    "http://10.0.0.248:19000",
    "http://10.0.0.248:19006",
]

# 2. Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,     # Allow cookies/auth headers
    allow_methods=["*"],         # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],         # Allow all request headers
)

# ML model for digit recognition
model = joblib.load("models/digit_svm.pkl")

def recognize_digits(cells):
    board = []
    for i in range(9):
        row = []
        for j in range(9):
            cell_img, is_empty = cells[i * 9 + j]
            if is_empty:
                row.append(".")
            else:
                cell_normalized = normalize_cell(cell_img)
                features = extract_features([cell_normalized])
                digit = model.predict(features)[0]
                row.append(str(digit))
        board.append(row)
    return board


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

            # Crop margin on all sides to exclude grid lines from the digit region
            margin_percent = 0.12
            margin_y = int(cell_h * margin_percent)
            margin_x = int(cell_w * margin_percent)
            cell_cropped = cell[margin_y:-margin_y, margin_x:-margin_x]

            # Use the thresholded cell (not the cropped one) for empty detection
            # since is_cell_empty applies its own internal margin crop
            isEmpty = is_cell_empty(thresh[y_start:y_end, x_start:x_end], threshold_percent=0.06, margin_percent=0.15)

            cells.append((cell_cropped, isEmpty))

    return cells  # 81 (cell_image, is_empty) tuples in row-major order


@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    # Read the uploaded image file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Pipeline
    try:
        if image.mean() < 127:
            image = cv2.bitwise_not(image) # Replace this with custom implementation
        original = image.copy()
        gaussian_kernel = create_gaussian_kernel(9, 1.6)
        blurred = linear_filter(image, gaussian_kernel, is_clipped=True)

        # Thresholding
        thresh = apply_adaptive_threshold(blurred, 11, 2, is_inverse=True)

        # Find contours (I don't think we can build custom version)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Suzuki Abe algorithm
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        grid_contour = None
        for contour in contours[:3]:
            contour = contour[:,0,:]
            perimeter = find_arc_length(contour, is_closed=True)
            epsilon = 0.02 * perimeter
            approx = approximate_polygon(contour, epsilon, is_closed=True)
            if len(approx) == 4:
                grid_contour = approx
                break

        if grid_contour is None:
            raise ValueError("Could not find 4 corners of the Sudoku board. Try better lighting.")


        # Perspective Transform (Flatten the board)
        rect = order_points(grid_contour)

        # Create a perfect square image to map the board to (450x450 is easily divisible by 9)
        side_length = 450
        dst = np.array([
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1]
        ], dtype="float32")

        # Calculate the homography matrix and apply it
        matrix = get_perspective_transform(rect, dst)

        # added logic to pick and choose the warp function and which one to use overall
        board_img = warp_perspective_inverse(original, matrix, (side_length, side_length))

        rotation_scores = {}
        for angle in [0, 90, 180, 270]:
            rotated = rotate_board(board_img, angle)
            cells = extract_sudoku_cells(rotated)
            non_empty = [c[0] for c in cells if not c[1]]
            if len(non_empty) == 0:
                continue
            confidences = [model.decision_function(extract_features([normalize_cell(c)]))[0].max() for c in non_empty]
            avg_confidence = np.mean(confidences)
            rotation_scores[angle] = avg_confidence
                
        rotation_scores = dict(sorted(rotation_scores.items(), key=lambda item: item[1], reverse=True))

        board = None
        for angle, score in rotation_scores.items():
            final_board_img = rotate_board(board_img, angle)
            final_cells = extract_sudoku_cells(final_board_img)
            board = recognize_digits(final_cells)
            if is_valid_sudoku(board):
                print(f"{angle} degrees is valid")
                break

        if board is None or not is_valid_sudoku(board):
            raise ValueError("Invalid board — digit recognition may have errors or the board is impossible to solve.")

        given = [[cell != "." for cell in row] for row in board]
        solve_sudoku(board)
        return {"status": "success", "board": board, "given": given}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))