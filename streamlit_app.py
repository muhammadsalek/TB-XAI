"""
TB-XAI Streamlit Application
Research prototype for:
1) single-vs-multiple annotated TB lesion-burden classification (ConvNeXtTiny),
2) Grad-CAM / Grad-CAM++ explanation,
3) TB lesion localization (YOLO26n).

IMPORTANT:
The Stage-8 classifier in the project notebook was trained with:
    0 = single annotated lesion
    1 = multiple annotated lesions
It is NOT a TB-vs-normal diagnostic classifier.
"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TB-XAI",
    page_icon="🫁",
    layout="wide",
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

APP_ROOT = Path(__file__).resolve().parent

YOLO_MODEL_PATH = APP_ROOT / "models" / "yolo26n_tb_best.pt"

# The ConvNeXt checkpoint is stored as a GitHub Release asset
# because it is too large for normal GitHub repository upload.
CLASSIFIER_RELEASE_URL = (
    "https://github.com/muhammadsalek/TB-XAI/releases/download/"
    "v1.0-models/ConvNeXtTiny_best.pth"
)

CLASSIFIER_CACHE_DIR = Path.home() / ".cache" / "tb_xai"
CLASSIFIER_CACHE_PATH = (
    CLASSIFIER_CACHE_DIR / "ConvNeXtTiny_best.pth"
)

# Stage 8/9-10 settings from the project notebook/results
CLASSIFIER_IMAGE_SIZE = 224
TEMPERATURE = 0.9915
CLASSIFIER_THRESHOLD = 0.4959

# Stage 11 validation-selected YOLO confidence threshold
YOLO_CONFIDENCE = 0.40
YOLO_IMAGE_SIZE = 640

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ]
)


# ============================================================
# MODEL DOWNLOAD / LOAD
# ============================================================

def ensure_classifier_checkpoint() -> Path:
    """Download the ConvNeXtTiny checkpoint once if not already cached."""

    if CLASSIFIER_CACHE_PATH.exists():
        return CLASSIFIER_CACHE_PATH

    CLASSIFIER_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    urllib.request.urlretrieve(
        CLASSIFIER_RELEASE_URL,
        CLASSIFIER_CACHE_PATH,
    )

    return CLASSIFIER_CACHE_PATH


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Rebuild the Stage-8 ConvNeXtTiny architecture and load its state dict."""

    checkpoint_path = ensure_classifier_checkpoint()

    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        1,
    )

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    # Handle a few common checkpoint wrappers if needed.
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    # Remove a DataParallel "module." prefix if present.
    if isinstance(state_dict, dict):
        state_dict = {
            key.replace("module.", "", 1)
            if key.startswith("module.")
            else key: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource(show_spinner=False)
def load_detector():
    """Load the final project YOLO26n lesion detector."""

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    return YOLO(
        str(YOLO_MODEL_PATH)
    )


# ============================================================
# CALIBRATION + CLASSIFICATION
# ============================================================

def calibrated_probability(
    logit: float,
    temperature: float = TEMPERATURE,
) -> float:
    """Temperature-scaled sigmoid probability."""

    scaled = np.clip(
        logit / temperature,
        -50.0,
        50.0,
    )

    return float(
        1.0 / (1.0 + np.exp(-scaled))
    )


def classify_image(
    model,
    pil_image: Image.Image,
):
    """Return calibrated probability and lesion-burden class."""

    input_tensor = (
        EVAL_TRANSFORM(pil_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logit = (
            model(input_tensor)
            .reshape(-1)[0]
            .item()
        )

    probability = calibrated_probability(
        logit,
        TEMPERATURE,
    )

    predicted_class = int(
        probability >= CLASSIFIER_THRESHOLD
    )

    label = (
        "Multiple annotated lesions"
        if predicted_class == 1
        else "Single annotated lesion"
    )

    return (
        probability,
        predicted_class,
        label,
        input_tensor,
    )


# ============================================================
# CUSTOM GRAD-CAM / GRAD-CAM++
# Adapted from the project notebook implementation
# ============================================================

class CustomCAM:
    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

        if output.requires_grad:
            output.register_hook(
                self._gradient_hook
            )

    def _gradient_hook(
        self,
        gradient,
    ):
        self.gradients = gradient

    def remove(self):
        self.forward_handle.remove()


def normalize_cam(
    cam: np.ndarray,
) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= cam.min()

    max_value = cam.max()

    if max_value > 0:
        cam /= max_value

    return cam


def generate_gradcam(
    model,
    input_tensor,
    target_class: int,
) -> np.ndarray:

    target_layer = model.features[-1]
    extractor = CustomCAM(
        model,
        target_layer,
    )

    try:
        model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():
            output = model(
                input_tensor
            ).reshape(-1)

            logit = output[0]

            score = (
                logit
                if target_class == 1
                else -logit
            )

            score.backward(
                retain_graph=False
            )

        activations = (
            extractor.activations
            .detach()
        )

        gradients = (
            extractor.gradients
            .detach()
        )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = F.relu(cam)

        cam = (
            cam[0]
            .cpu()
            .numpy()
        )

        return normalize_cam(cam)

    finally:
        extractor.remove()


def generate_gradcam_pp(
    model,
    input_tensor,
    target_class: int,
) -> np.ndarray:

    target_layer = model.features[-1]
    extractor = CustomCAM(
        model,
        target_layer,
    )

    try:
        model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():
            output = model(
                input_tensor
            ).reshape(-1)

            logit = output[0]

            score = (
                logit
                if target_class == 1
                else -logit
            )

            score.backward(
                retain_graph=False
            )

        activations = (
            extractor.activations
            .detach()
        )

        gradients = (
            extractor.gradients
            .detach()
        )

        gradients_2 = gradients ** 2
        gradients_3 = gradients ** 3

        spatial_sum = (
            activations * gradients_3
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        denominator = (
            2 * gradients_2
            + spatial_sum
            + 1e-8
        )

        alpha = (
            gradients_2
            / denominator
        )

        positive_gradients = F.relu(
            gradients
        )

        weights = (
            alpha * positive_gradients
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = F.relu(cam)

        cam = (
            cam[0]
            .cpu()
            .numpy()
        )

        return normalize_cam(cam)

    finally:
        extractor.remove()


def overlay_heatmap(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:

    rgb = np.asarray(
        pil_image.convert("RGB")
    )

    height, width = rgb.shape[:2]

    resized_cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.uint8(
        255 * resized_cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        rgb,
        1.0 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay


# ============================================================
# YOLO LOCALIZATION
# ============================================================

def run_yolo(
    detector,
    pil_image: Image.Image,
):
    """Run YOLO26n and return the rendered image + detection count."""

    results = detector.predict(
        source=np.asarray(
            pil_image.convert("RGB")
        ),
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=0.70,
        max_det=100,
        verbose=False,
    )

    result = results[0]

    number_of_detections = (
        0
        if result.boxes is None
        else len(result.boxes)
    )

    plotted_bgr = result.plot(
        labels=True,
        conf=True,
    )

    plotted_rgb = cv2.cvtColor(
        plotted_bgr,
        cv2.COLOR_BGR2RGB,
    )

    confidences = []

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):
        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

    return (
        plotted_rgb,
        number_of_detections,
        confidences,
    )


# ============================================================
# UI
# ============================================================

st.title(
    "🫁 TB-XAI"
)

st.caption(
    "Explainable chest X-ray lesion-burden classification "
    "and TB lesion localization"
)

st.warning(
    "Research prototype only. The ConvNeXtTiny classifier in this "
    "project predicts single vs multiple annotated lesions; it is NOT "
    "a TB-vs-normal diagnostic classifier and must not be used for "
    "clinical diagnosis or medical decision-making."
)

with st.expander(
    "Model settings",
    expanded=False,
):
    st.write(
        f"Classifier: ConvNeXtTiny | input {CLASSIFIER_IMAGE_SIZE}×"
        f"{CLASSIFIER_IMAGE_SIZE} | temperature={TEMPERATURE:.4f} | "
        f"decision threshold={CLASSIFIER_THRESHOLD:.4f}"
    )
    st.write(
        f"Detector: YOLO26n | input {YOLO_IMAGE_SIZE} | "
        f"confidence threshold={YOLO_CONFIDENCE:.2f}"
    )
    st.write(
        f"Runtime device: {DEVICE}"
    )

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)

if uploaded_file is None:
    st.info(
        "Upload a PNG or JPEG chest X-ray to run the research pipeline."
    )
    st.stop()


try:
    image = Image.open(
        uploaded_file
    ).convert("RGB")
except Exception as exc:
    st.error(
        f"Could not read the uploaded image: {exc}"
    )
    st.stop()


st.subheader(
    "Input"
)

st.image(
    image,
    caption="Uploaded chest X-ray",
    width="stretch",
)


# ============================================================
# LOAD MODELS
# ============================================================

try:
    with st.spinner(
        "Loading ConvNeXtTiny classifier..."
    ):
        classifier = load_classifier()
except Exception as exc:
    st.error(
        "The ConvNeXtTiny classifier could not be loaded. "
        "Check the GitHub Release asset and internet access."
    )
    st.exception(exc)
    st.stop()


try:
    with st.spinner(
        "Loading YOLO26n detector..."
    ):
        detector = load_detector()
except Exception as exc:
    st.error(
        "The YOLO26n detector could not be loaded. "
        "Make sure models/yolo26n_tb_best.pt exists in the repository."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# RUN CLASSIFIER
# ============================================================

with st.spinner(
    "Running ConvNeXtTiny..."
):
    (
        probability,
        predicted_class,
        class_label,
        input_tensor,
    ) = classify_image(
        classifier,
        image,
    )


st.subheader(
    "Lesion-burden classification"
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Predicted class",
    class_label,
)

m2.metric(
    "Calibrated P(multiple lesions)",
    f"{probability:.3f}",
)

m3.metric(
    "Decision threshold",
    f"{CLASSIFIER_THRESHOLD:.4f}",
)

st.caption(
    "Class definition from the project notebook: "
    "0 = single annotated lesion; 1 = multiple annotated lesions."
)


# ============================================================
# XAI
# ============================================================

st.subheader(
    "Explainable AI"
)

try:
    with st.spinner(
        "Generating Grad-CAM and Grad-CAM++..."
    ):
        gradcam = generate_gradcam(
            classifier,
            input_tensor,
            predicted_class,
        )

        gradcam_pp = generate_gradcam_pp(
            classifier,
            input_tensor,
            predicted_class,
        )

        gradcam_overlay = overlay_heatmap(
            image,
            gradcam,
        )

        gradcam_pp_overlay = overlay_heatmap(
            image,
            gradcam_pp,
        )

    x1, x2 = st.columns(2)

    with x1:
        st.image(
            gradcam_overlay,
            caption="Grad-CAM",
            width="stretch",
        )

    with x2:
        st.image(
            gradcam_pp_overlay,
            caption="Grad-CAM++",
            width="stretch",
        )

except Exception as exc:
    st.warning(
        "XAI visualization could not be generated for this image."
    )
    st.exception(exc)


# ============================================================
# YOLO
# ============================================================

st.subheader(
    "YOLO26n TB lesion localization"
)

try:
    with st.spinner(
        "Running YOLO26n lesion detector..."
    ):
        (
            yolo_image,
            detection_count,
            confidences,
        ) = run_yolo(
            detector,
            image,
        )

    y1, y2 = st.columns(
        [3, 1]
    )

    with y1:
        st.image(
            yolo_image,
            caption=(
                "YOLO26n predicted lesion boxes "
                f"(confidence ≥ {YOLO_CONFIDENCE:.2f})"
            ),
            width="stretch",
        )

    with y2:
        st.metric(
            "Detected lesion boxes",
            detection_count,
        )

        if confidences:
            st.write(
                "Detection confidence:"
            )

            for index, confidence in enumerate(
                confidences,
                start=1,
            ):
                st.write(
                    f"Box {index}: {confidence:.3f}"
                )
        else:
            st.write(
                "No lesion box exceeded the selected confidence threshold."
            )

except Exception as exc:
    st.warning(
        "YOLO localization could not be generated for this image."
    )
    st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TB-XAI is an educational/research project. "
    "Outputs are experimental and are not medical advice."
)
