
# 🔄 Self-Improving Object Detection System

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.4-green)]()
[![Stable Diffusion](https://img.shields.io/badge/SD-v1.5-purple)]()
[![COCO](https://img.shields.io/badge/Dataset-COCO2017-orange)]()

A deep learning system that **automatically detects its own
weaknesses and generates synthetic training data to fix them**,
achieving a **+29.3% mAP improvement** without manual data
collection.

---

## 📊 Results

| Metric | Baseline | Retrained | Improvement |
|--------|----------|-----------|-------------|
| mAP@50 | 0.581 | **0.752** | +29.3% |
| mAP@50-95 | 0.451 | **0.632** | +40.4% |
| Precision | 0.611 | **0.791** | +29.6% |
| Recall | 0.466 | **0.662** | +42.0% |

---

## 🔄 How It Works
```
Train YOLOv8n → Detect Failures → DINO Anomalies
      ↑                                   ↓
  Retrain              Stable Diffusion Generates
      ↑                  Synthetic Hard Images
      └──────────────────────────────────────────
```

1. **Phase 1** — Train YOLOv8n baseline on COCO 2017
2. **Phase 2** — Failure detection using confidence + entropy
3. **Phase 3** — DINO embeddings find anomalous images
4. **Phase 4** — Stable Diffusion generates hard scenarios
5. **Phase 5** — Expand dataset with synthetic images
6. **Phase 6** — Retrain and measure improvement

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| YOLOv8n | Object detection backbone |
| DINO ViT-S/8 | Self-supervised anomaly detection |
| Stable Diffusion v1.5 | Synthetic image generation |
| PyTorch 2.9 | Deep learning framework |
| COCO 2017 | Training dataset (80 classes) |
| Streamlit | Web app deployment |
| Kaggle T4 GPU | Training infrastructure |

---

## 📁 Project Structure
```
self_improving_detector/
├── src/
│   ├── failure_detector.py     # Phase 4
│   └── embedding_engine.py     # Phase 5
├── data/
│   ├── processed/              # COCO 2017 subset
│   ├── synthetic/              # Generated images
│   └── expanded/               # Merged dataset
├── models/
│   └── checkpoints/
│       ├── yolov8n_fast_best.pt
│       └── yolov8n_retrained_best.pt
├── outputs/
│   ├── charts/                 # All visualization
│   └── predictions/            # Model outputs
└── app/
    └── app.py                  # Streamlit web app
```

---

## 🚀 Quick Start
```bash
# Install dependencies
pip install ultralytics diffusers transformers
pip install timm streamlit pyngrok

# Run web app
streamlit run app/app.py
```

---

## 📈 Training Details

- **Dataset**: COCO 2017 (128 train + 64 val images)
- **Model**: YOLOv8n (3.2M parameters)
- **Epochs**: 25
- **Batch size**: 32
- **Device**: Kaggle T4 GPU (15.6 GB)
- **Synthetic data**: 20 images, 5 scenarios

### Hard Scenarios Generated:
- 🌫️ Foggy roads
- 🌙 Night streets
- 🌧️ Rainy intersections
- 👥 Crowded markets
- 🏗️ Construction zones


## 📞 Author

Built as a deep learning lab project demonstrating
the complete MLOps pipeline from data preparation
to deployment.
