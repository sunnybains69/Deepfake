# Deepfake
Deepfake Detection program

EfficientNet-B0 face-based classifier (PyTorch), with a Streamlit app (`app.py`).

## Retraining workflow (leakage-free)

Run every command from the project root.

### 1. Build a video-level split

Every video, and by default every person, goes into exactly one of
train / val / test, so the model cannot memorise faces it is then
"validated" on. Faces are cropped with the same function the app uses.

```bash
python -m src.prepare_split \
    --real-dir FF++/original_sequences/youtube/c23/videos \
    --fake-dir FF++/manipulated_sequences/Deepfakes/c23/videos \
    --fake-dir FF++/manipulated_sequences/FaceSwap/c23/videos \
    --output-dir data_by_video \
    --frames-per-video 20
```

Repeat `--fake-dir` for each manipulation method you use. The output
folder must be new or empty. `data_by_video/split_manifest.json`
records which video went where.

### 2–3. Train

Class weights, a learning-rate scheduler and early stopping are on by
default. The best model is chosen on validation loss.

```bash
python -m src.train --data-dir data_by_video --model-output models/best_model_faces.pth
```

Expect lower validation scores than before. Those are the honest numbers.

### 4. Calibrate the video threshold and test

```bash
python -m src.calibrate_threshold \
    --manifest data_by_video/split_manifest.json \
    --model-path models/best_model_faces.pth
```

This picks the threshold on validation videos, reports results on the
held-out test videos, and saves `models/threshold.json`, which
`predict_video.py` and the app load automatically. Re-run it after
every retrain.

### Predict

```bash
python -m src.predict_video --video path/to/video.mp4
streamlit run app.py
```

By default a video's score is the mean of its most suspicious 25% of
frames (`--aggregation topk --top-fraction 0.25`).
