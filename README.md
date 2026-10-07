# AI-based Plant Disease Detection

Graduation project: detect plant diseases from leaf images using Deep Learning, with a full-stack web application in Arabic.

## Features

- Plant disease classification from a single leaf photo (38 classes)
- Image validation pipeline: blur detection, plant-color filtering, non-plant rejection, confidence threshold
- Severity assessment and treatment recommendations in Arabic
- User authentication (register / login / password recovery)
- Personal dashboard with scan history and statistics
- Disease search with treatment advice

## Tech Stack

| Layer | Technologies |
|---|---|
| AI / ML | TensorFlow, MobileNetV2 (Transfer Learning + Fine-Tuning), Keras |
| Image Processing | OpenCV, Pillow, NumPy |
| Backend | Python, Flask, SQLite |
| Frontend | HTML5, CSS3, JavaScript |

## Model

- **Architecture:** MobileNetV2 pretrained on ImageNet, fine-tuned on the last 40 layers
- **Input:** 224x224x3
- **Training:** Data augmentation (flip, rotation, zoom, contrast), Adam optimizer (lr=1e-4), EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Dataset:** Plant Diseases Dataset (train / valid split)

## Project Structure

```
.
├── backend/
│   ├── app.py                  # Flask application + prediction API
│   ├── train.py                # Model training / fine-tuning script
│   ├── predict.py              # Standalone prediction helper
│   ├── classes.json            # Class index mapping
│   └── plant_disease_model.h5  # Trained model
├── frontend/
│   ├── templates/              # Landing, auth, predict, dashboard, history, search, FAQ
│   └── static/                 # CSS + JS
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt
python backend/app.py
```

Then open `http://localhost:5000`.

To retrain the model from scratch, place the dataset in `Plant Diseases Dataset/` with `train/` and `valid/` subfolders and run:

```bash
python backend/train.py
```

## Notes

- The dataset folder is not included in this repository due to its size.
- The repository is kept private as a graduation project submission.
