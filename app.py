from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.model import load_model
from src.predict import predict_pil_image
from src.predict_video import predict_video
from src.utils import get_device


MODEL_PATH = "models/best_model.pth"


st.set_page_config(
    page_title="Sunny's Deepfake Detection Program",
    page_icon="🔎",
layout="wide",
initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Project Information")

    st.write("**Researcher:** Sangeet Bains")
    st.write("**Programme:** MSc Cyber Security and Digital Forensics")
    st.write("**Institution:** University of Westminster")
    st.write("**Model:** EfficientNet-B0")
    st.write("**Output:** Real or Fake")

    st.divider()

    st.subheader("Analysis Settings")

    sample_every = st.slider(
        "Analyse every nth frame",
        min_value=1,
        max_value=60,
        value=15,
        help="Lower values analyse more frames but take longer.",
    )

    max_frames = st.slider(
        "Maximum frames to analyse",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
    )

    st.divider()

    st.warning(
        "This application is a research prototype. "
        "It's predictions should not be treated as forensic proof."
    )


# -------------------------------------------------
# Page Heading
# -------------------------------------------------



st.title("AI-Based Deepfake Detection Program")
st.subheader("MSc Cyber Security and Forensics")
st.caption(
    "Developed by Sunny Bains - University of Westminster."
)

st.info(
"Upload an image or video containing a visible face. "
"The system will estimate whether the media is authentic "
"or potentially manipulated."
)


# ------------------------------------------------
# Check Model
# ------------------------------------------------


model_file = Path(MODEL_PATH)

if not model_file.exists():
    st.error(
        "No trained model was found at models/best_model.pth. "
        "Train the model before running the interface."
    )
    st.stop()


# -----------------------------------------------
# Load model
# -----------------------------------------------


device = get_device()

model, checkpoint = load_model(
    str(model_file),
    device,
)

class_to_idx = checkpoint["class_to_idx"]


# ------------------------------------------------
# Select media
# ------------------------------------------------


device = get_device()
model, checkpoint = load_model(str(model_file), device)
class_to_idx = checkpoint["class_to_idx"]

mode = st.radio(
    "Select media type",
    ["Image", "Video"],
    horizontal=True,
)

# ------------------------------------------------
# Image analysis
# ------------------------------------------------


if mode == "Image":
    uploaded = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "webp"],
	help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image")

        result = predict_pil_image(
            image,
            model,
            class_to_idx,
            device,
        )

        st.subheader(
            f"Prediction: {result['prediction'].upper()}"
        )
        st.metric(
            "Fake probability",
            f"{result.get('fake', 0.0) * 100:.2f}%",
        )
        st.metric(
            "Real probability",
            f"{result.get('real', 0.0) * 100:.2f}%",
        )

else:
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_file.write(uploaded.read())
            temporary_path = temporary_file.name

        st.video(uploaded)

        with st.spinner("Analysing sampled frames..."):
            result = predict_video(
                video_path=temporary_path,
                model_path=MODEL_PATH,
                sample_every=15,
                max_frames=60,
            )

        st.subheader(
            f"Prediction: {result['prediction'].upper()}"
        )
        st.metric(
            "Fake probability",
            f"{result['fake_probability'] * 100:.2f}%",
        )
        st.metric(
            "Real probability",
            f"{result['real_probability'] * 100:.2f}%",
        )
        st.write(
            f"Frames sampled: {result['sampled_frames']}"
        )

st.divider()
st.write(
    "Limitations: performance depends on the training data and may "
    "degrade on unseen manipulation methods, compressed media, "
    "different demographics, or non-face content."
)
