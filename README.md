# Real-Time Hand Gesture Recognition

Real-time hand gesture recognition from a webcam using MediaPipe, OpenCV, and a Random Forest classifier. The application detects one hand, converts its landmarks into normalized numeric features, and displays one of five predicted gesture labels on the video feed.

## Features

- Real-time webcam inference
- MediaPipe hand detection and landmark tracking
- Position-normalized 42-feature representation
- Random Forest gesture classification
- Scripts for collecting a private dataset and training a new model
- Reproducible train/test split and classifier configuration

## Recognized gestures

| Class | Gesture |
| --- | --- |
| 0 | OK |
| 1 | PEACE |
| 2 | I LOVE YOU |
| 3 | HI |
| 4 | HIT ME UP |

## How it works

`webcam → MediaPipe hand landmarks → normalized 42 features → Random Forest → predicted gesture`

MediaPipe returns 21 two-dimensional hand landmarks. The minimum x and y coordinates are subtracted from each point to reduce sensitivity to hand position, producing 42 features. A scikit-learn Random Forest then predicts the gesture class.

## Technology stack

- Python 3.10
- OpenCV
- MediaPipe
- NumPy
- scikit-learn Random Forest

## Project structure

```text
.
├── collect_imgs.py           # Capture a private five-class image dataset
├── create_dataset.py         # Convert images to normalized landmark features
├── train_classifier.py       # Train and evaluate the Random Forest model
├── inference_classifier.py   # Run real-time webcam inference
├── requirements.txt          # Python dependencies
├── model.p                   # Pretrained model for immediate inference
├── LICENSE                   # MIT license from the original project
└── README.md
```

The training images and generated `data.pickle` feature dataset are intentionally excluded from the repository.

## Requirements

- Windows 11 was used for testing
- Python 3.10 is recommended
- A working webcam

## Installation

### PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, either adjust the execution policy for the current process or run the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### General terminals

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt, activate with `.venv\Scripts\activate.bat`.

## Run inference

With the virtual environment activated:

```bash
python inference_classifier.py
```

The default webcam is camera index `0`. To close the webcam cleanly, focus the window named `frame` and press the lowercase `q` key.

## Collect your own dataset and retrain

Dataset collection is optional because a pretrained `model.p` is included. To build a model from your own images:

```bash
python collect_imgs.py
python create_dataset.py
python train_classifier.py
```

`collect_imgs.py` creates five class folders and captures 100 images per class. For each class, pose your hand in view and press lowercase `q` to begin capture. `create_dataset.py` extracts the 42 landmark features, and `train_classifier.py` creates a new `model.p` using a reproducible random seed.

## Model performance

The included workflow achieved **100% accuracy on one held-out 20% test split** of the self-collected dataset. The dataset is small and contains closely related webcam frames, so this result can be optimistic. It does not establish real-world accuracy or generalization to different people, cameras, environments, or signing styles.

## Limitations

- Recognizes only five static gestures
- Processes one hand at a time
- Can be sensitive to lighting, background, camera angle, and hand placement
- Is a gesture-classification demonstration, not a complete sign-language translator

## Dataset privacy

The self-collected webcam images are intentionally excluded to protect privacy. Users who want to retrain the classifier can create their own local dataset with `collect_imgs.py`. The `data/` directory and derived `data.pickle` file are ignored by Git.

## Security note

Python pickle files can execute code when loaded. Only use `model.p` or `data.pickle` files from sources you trust. The included model was regenerated locally from the private training dataset during project validation.

## Attribution

The core workflow in this implementation was adapted from the Computer Vision Engineer sign-language detector tutorial and repository:

- Original repository: [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python)
- Original YouTube tutorial: [Sign language detection with Python and Scikit Learn](https://www.youtube.com/watch?v=MJCSjXepaAM)

This portfolio version retains that workflow rather than claiming the underlying tutorial code as independently invented. Additions made for this implementation include five gesture classes, improved webcam and resource handling, modernized dependency pins, dataset and model validation, reproducible training settings, and a regenerated model.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE). The original copyright and permission notice are preserved.
