
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.model import load_model
from src.predict import predict_pil_image
from src.predict_video import predict_video
from src.utils import get_device


# New face-based EfficientNet-B0 model
MODEL_PATH = "models/best_model_faces.pth"


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
    st.write("**Model:** EfficientNet-B0 (Face-Based)")
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
        "Its predictions should not be treated as forensic proof."
    )


# --------------------------------------------------
# Page Heading
# --------------------------------------------------

st.title("AI-Based Deepfake Detection Program")
st.subheader("MSc Cyber Security and Digital Forensics")

st.caption(
    "Developed by Sunny Bains - University of Westminster."
)

st.info(
    "Upload an image or video containing a visible face. "
    "The system will detect and crop the face before estimating "
    "whether the media is authentic or potentially manipulated."
)


# --------------------------------------------------
# Check Model
# --------------------------------------------------

model_file = Path(MODEL_PATH)

if not model_file.exists():
    st.error(
        f"No trained model was found at {MODEL_PATH}. "
        "Check that the model file exists."
    )
    st.stop()


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def get_model(model_path):
    device = get_device()
    model, checkpoint = load_model(model_path, device)
    model.eval()
    return model, checkpoint, device


model, checkpoint, device = get_model(str(model_file))
class_to_idx = checkpoint["class_to_idx"]


# --------------------------------------------------
# Select Media
# --------------------------------------------------

mode = st.radio(
    "Select media type",
    ["Image", "Video"],
    horizontal=True,
)


# --------------------------------------------------
# Image Analysis
# --------------------------------------------------

if mode == "Image":

    uploaded = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
        )

        with st.spinner("Detecting face and analysing image..."):
            try:
                result = predict_pil_image(
                    image,
                    model,
                    class_to_idx,
                    device,
                    crop_faces=True,
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()

        st.subheader(
            f"Prediction: {result['prediction'].upper()}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fake probability",
                f"{result['fake'] * 100:.2f}%",
            )

        with col2:
            st.metric(
                "Real probability",
                f"{result['real'] * 100:.2f}%",
            )



# --------------------------------------------------
# Video Analysis
# --------------------------------------------------

else:
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded is not None:
        st.video(uploaded)

        suffix = Path(uploaded.name).suffix
        temporary_path = None
        result = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(uploaded.getvalue())
                temporary_path = temporary_file.name

            with st.spinner(
                "Detecting faces and analysing video frames..."
            ):
                result = predict_video(
                    video_path=temporary_path,
                    model_path=MODEL_PATH,
                    sample_every=sample_every,
                    max_frames=max_frames,
                    crop_faces=True,
                    model=model,
                    checkpoint=checkpoint,
                    device=device,
                )

        except (ValueError, RuntimeError) as error:
            st.error(f"Video analysis failed: {error}")

        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

        if result is not None:
            prediction = result["prediction"].upper()

            if prediction == "UNCERTAIN":
                st.warning(
                    "Prediction: UNCERTAIN - the model cannot "
                    "reliably determine whether this video "
                    "is REAL or FAKE."
                )
            elif prediction == "FAKE":
                st.error("Prediction: FAKE")
            elif prediction == "REAL":
                st.success("Prediction: REAL")
            else:
                st.warning(f"Prediction: {prediction}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Video fake score",
                    f"{result['fake_probability'] * 100:.2f}%",
                )

            with col2:
                st.metric(
                    "Decision threshold",
                    f"{result['threshold'] * 100:.2f}%",
                )

            if not result["threshold_calibrated"]:
                st.caption(
                    "Threshold is the uncalibrated default (50%). "
                    "Run calibrate_threshold.py for a validated one."
                )

            st.write(
                f"Frames analysed: {result['sampled_frames']}"
            )

            st.write(
                "Frames skipped (no face detected): "
                f"{result['skipped_faces']}"
            )

            st.line_chart(
                result["frame_fake_probabilities"],
                x_label="Analysed frame",
                y_label="Fake probability",
            )

# --------------------------------------------------
# Limitations
# --------------------------------------------------

st.divider()

st.write(
    "Limitations: This research prototype may incorrectly classify "
    "genuine or manipulated media. Performance depends on the "
    "training dataset, face visibility, image quality, compression, "
    "and manipulation method. Model probabilities are not "
    "calibrated forensic confidence scores."
)
