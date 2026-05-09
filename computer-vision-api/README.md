# Optical-Sudoku-Solver
Solves Sudoku puzzles using computer vision

## Installing dependencies
Create environment: `python -m venv venv`

Activate it:
- Windows: `.\venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

Install your requirements: `pip install -r requirements.txt`

To deactivate virtual environment: `deactivate`

### Running Jupyter Notebook and Linking Venv to Jupyter

After installing dependencies, you must register the virtual environment as a Jupyter kernel:
1. `python -m ipykernel install --user --name=sudoku-env --display-name "Python (Sudoku Project)"`
2. Run: `jupyter notebook`
3. In your notebook, go to **Kernel -> Change Kernel** and select **"Python (Sudoku Project)"**.


## Dataset

### Sudoku Dataset
The [Sudoku Dataset](https://github.com/wichtounet/sudoku_dataset) repository is used in `EvaluationSVM.ipynb`, `DigitRecognitionCNN.ipynb`, and `DigitRecognitionCNN_ran_on_Colab.ipynb`.

Consider cloning the [Sudoku Dataset](https://github.com/wichtounet/sudoku_dataset) repository and placing it in the `data/` directory

### Digit Recognition Dataset
We use the [Chars74K EnglishFnt](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) subset — computer-rendered digits across 1,016 fonts, well-matched to printed Sudoku puzzles.

**No manual download required.** Run `DigitRecognitionSVM.ipynb` and the first cell will automatically download and extract the dataset into `data/`.


## Usage

### Digit recognition models

There are two digit recognition models
- SVM (Support Vector Machines)
- CNN (Convolutional Neural Network)

#### Training the SVM model
To train the SVM model, run `DigitRecognitionSVM.ipynb` — this downloads the required datasets, trains the SVM, and saves the model to `models/digit_svm.pkl`.

> **Note:** `models/digit_svm.pkl` is not included in the repository (file size exceeds GitHub's 100MB limit). You must generate it by running `DigitRecognitionSVM.ipynb` before using `main.ipynb`.

#### To evaluate SVM's digit recognition performance
Run `EvaluationSVM.ipynb` after `DigitRecognitionSVM.ipynb`. Requires `models/digit_svm.pkl` and `data/test_puzzles.txt` to exist.

#### Training the CNN model

To train the CNN model, run `DigitRecognitionCNN.ipynb` — this downloads the required datasets, trains the CNN, and saves the model to `models/digit_cnn.pth`. 

> **Note:** Training the CNN model is GPU intensive and may take a very long time. If your GPU is not powerful, consider copying over `DigitRecognitionCNN_ran_on_Colab.ipynb` (which contains all the dependencies) onto Google Colab and use the GPU runtime provided by Google.


### Running the Pipeline
Once the digit recognition models have been trained, you should see the following files: `models/digit_cnn.pth` and `models/digit_svm.pkl`.

Run `main.ipynb` to run the sudoku solver pipeline.