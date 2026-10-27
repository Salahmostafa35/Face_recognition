# Face Recognition with Face Embeddings and KNN

A Jupyter notebook project for identifying known people in images. It uses the `face_recognition` library to detect faces and extract face encodings, then classifies those encodings with a distance-weighted k-nearest neighbors (KNN) model from scikit-learn. The notebook also evaluates the classifier and demonstrates an `unknown` label for faces that are too far from the nearest training example.

## What it does

- Explores the included face dataset and label distribution.
- Detects faces in images and extracts one encoding for each image containing exactly one face.
- Trains a KNN classifier (`n_neighbors=2`, `weights="distance"`) to assign one of the known identity labels.
- Evaluates predictions with a classification report and confusion matrix.
- Runs a sample-image recognition flow that labels faces as a known person or `unknown` based on a distance threshold.

This repository is a notebook-based experiment; it does not currently include a web app, command-line interface, saved trained model, or automated test suite.

## Repository contents

```text
.
├── face-recognation.ipynb       # Main exploration, training, evaluation, and demo notebook
├── Dataset.csv                  # Image filename and identity labels used for a label-count view
├── Faces/Faces/                 # Labeled face-image dataset
└── Original Images/Original Images/
                                # Original identity folders used by the notebook
```

The notebook trains from `Original Images/Original Images/`. It reads `Dataset.csv` for its initial label exploration. The `Faces/Faces/` directory contains the separate labeled face-image set represented by that CSV.

## Requirements

- Python 3.9 or newer is recommended.
- Jupyter Notebook or JupyterLab.
- The `face_recognition` package and its `dlib` dependency. Installing `dlib` may require a C++ build toolchain and CMake, depending on your operating system and Python version.
- Python packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`, `scikit-learn`, and `tqdm`.

Create and activate an environment, then install the project dependencies:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install cmake face-recognition numpy pandas matplotlib seaborn opencv-python scikit-learn tqdm jupyter
```

`face_recognition` relies on `dlib`; installation can vary across platforms. If installation fails, follow the package's installation guidance for your OS and Python version before continuing.

## Run the notebook

1. Clone or download this repository and install the dependencies above.
2. Open `face-recognation.ipynb` in Jupyter.
3. Update the paths in the notebook's dataset-loading cells to point to this checkout. The current notebook uses absolute Windows paths such as `D:\Face recognition\Dataset.csv` and `D:\Face recognition\Original Images\Original Images`.
4. Run the notebook cells in order.

For example, define paths relative to the notebook instead of using machine-specific paths:

```python
from pathlib import Path

project_dir = Path.cwd()
csv_path = project_dir / "Dataset.csv"
image_dir = project_dir / "Original Images" / "Original Images"
```

Then use `pd.read_csv(csv_path)` and `str(image_dir)` where the notebook currently uses hard-coded paths. Run Jupyter with the repository root as its working directory so these relative paths resolve as expected.

The notebook processes images with `face_recognition.face_locations` and `face_recognition.face_encodings`. Only images with exactly one detected face are retained for training. The checked-in notebook reports 2,499 usable face encodings, then uses an 80/20 stratified train/test split (`random_state=42`).

## Recognition approach

1. **Face detection:** Find face locations in each image.
2. **Encoding:** Convert each single detected face to a numeric face embedding using `face_recognition`.
3. **Classification:** Fit a distance-weighted KNN classifier with two neighbors on the training embeddings.
4. **Unknown handling:** In the demo cell, compare the nearest-neighbor distance to a threshold of `0.6`. A face beyond this threshold is labeled `unknown`; otherwise, the KNN model predicts one of the identities represented in the training data.

The notebook's sample evaluation includes 31 identity classes and 500 test images. Its classification report displays an overall accuracy of `1.00` at the printed precision. This is the notebook's result on its own image-level split, not a guarantee of performance on new images or people. Images of the same identity may share capture or source characteristics across the split, so independent evaluation is needed before drawing broader conclusions.

## Dataset and intended use

The repository contains images labeled with public-figure names. The notebook uses those names as identity classes; it does not support arbitrary people unless you add appropriately labeled reference images and retrain the classifier. Dataset provenance and reuse terms are not documented in this repository, so review applicable image rights and dataset terms before redistribution or deployment.

Face recognition is sensitive biometric processing. Use this project only with appropriate consent and a lawful, clearly defined purpose. The sample notebook is an educational experiment and should not be used for consequential identity decisions, access control, surveillance, or other high-impact decisions.

## Known limitations

- The notebook contains absolute Windows file paths that must be changed for your environment.
- The `unknown` threshold is a fixed distance cutoff and has not been calibrated for general use.
- The evaluation uses a random image-level split from this dataset and may overstate performance on unrelated images or real-world camera conditions.
- The notebook does not persist the fitted KNN model; rerunning the training cells is required to use the recognition demo in a fresh session.
- Face detection or encoding may fail or skip images with no face, multiple faces, poor lighting, pose, or low image quality.

## License

No license file is included in the repository. Unless the owner adds a license, standard copyright restrictions apply to reuse and redistribution of the code. Dataset and image rights may have separate terms.
