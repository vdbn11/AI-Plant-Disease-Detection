# AI-Based Plant Disease Detection

An end-to-end web application that identifies plant diseases from leaf photos and returns the diagnosis,
a confidence score and an Arabic treatment guide. Built with **Flask**, **TensorFlow/Keras**, **OpenCV** and **SQLite**.

---

## Highlights

- **58 disease classes** across 14+ crops (apple, banana, corn, grape, mango, orange, peach, pepper, potato,
  rice, soybean, squash, strawberry, cherry, blueberry, raspberry, tomato).
- **97.2% validation accuracy** on a held-out validation split (58-class model).
- **Arabic-first interface** with treatment recommendations for every class.
- Multi-user system with authentication, scan history and an analytics dashboard.

---

## Features

| Feature | Description |
|---|---|
| Leaf prediction | Upload a leaf image and receive the disease class with a confidence score |
| Smart cropping | OpenCV pre-processing crops and normalizes the leaf before inference |
| Accounts | Register, login, logout and password reset, with server-side sessions |
| History | Every scan is stored per user and can be reviewed or deleted |
| Dashboard | Totals and statistics for the logged-in user's scans |
| Disease guide | Before/after leaf images, symptoms and Arabic treatment advice for all classes |
| Search | Search the disease catalogue by crop or disease name |
| FAQ | Frequently asked questions page |

---

## Model

| Item | Value |
|---|---|
| Architecture | Convolutional neural network (TensorFlow / Keras) |
| Input size | 192 × 192 RGB |
| Classes | 58 |
| Validation accuracy | 97.19% |
| Validation loss | 0.087 |
| Training time | ~214 minutes |

Training details and the full class list are stored in `backend/training_report_v6.json` and
`backend/classes_v6.json`.

---

## Tech stack

- **Backend:** Python, Flask, TensorFlow / Keras, OpenCV, NumPy
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript (Jinja templates)

---

## Project structure

```
backend/
  app.py                     # Flask application (routes, auth, prediction, dashboard)
  train.py                   # Training script
  classes_v6.json            # 58 class names
  guide_data.json            # Arabic disease guide (symptoms, treatment, images)
  guide_manifest.json        # Guide image manifest
  training_report_v6.json    # Training metrics
  plant_disease_model_v6.keras
frontend/
  templates/                 # Jinja templates (landing, login, predict, history, dashboard, guide, search, faq)
  static/                    # CSS, JS and before/after guide images
requirements.txt
run_app.bat
```

---

## Run locally

```bash
pip install -r requirements.txt
python backend/app.py
```

Then open <http://127.0.0.1:5000>. On Windows you can also double-click `run_app.bat`.

The application creates `backend/database/plant_records.db` and `backend/uploads/` automatically on first run.

---

## Notes

- The training dataset and user uploads are **not** included in this repository.
- The model is provided so the application can be run and evaluated without retraining.
- `backend/train.py` documents how the model was trained.
