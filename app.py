"""Streamlit interface for local facial-expression inference."""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from PIL import Image, ImageDraw, ImageOps
from torch.nn import functional as F

from face_detection import FaceDetector, FaceScoreSmoother, crop_face
from model import (
    EMOTIONS,
    IMAGE_SIZE,
    ScoreOptions,
    expression_scores,
    get_score_options,
    load_checkpoint,
    preprocess_face,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCENT = "#ed906d"
INK = "#f4ebdc"
LINE = "#39302d"
ACCENT_RGB = (237, 144, 109)
INK_RGB = (244, 235, 220)
LABEL_RGB = (28, 23, 22)

st.set_page_config(page_title="ExpressionLab | Lab", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root {
  --lab-bg: #101011;
  --lab-panel: #1b1717;
  --lab-line: #39302d;
  --lab-ink: #f4ebdc;
  --lab-muted: #b9aaa2;
  --lab-accent: #ed906d;
  --lab-focus: #ffb08c;
}
.stApp { background: var(--lab-bg); color: var(--lab-ink); }
.stApp, .stApp h1, .stApp h2, .stApp h3 {
  font-family: "Segoe UI", system-ui, sans-serif;
}
.stApp ::selection { background: #70402f; color: var(--lab-ink); }
[data-testid="stHeader"] { background: var(--lab-bg); }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stSidebar"] { background: var(--lab-panel); border-right: 1px solid var(--lab-line); }
[data-testid="stMainBlockContainer"] { max-width: 1328px; padding: 4rem clamp(1.25rem, 5vw, 4.5rem) 3rem; }
.lab-masthead {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  min-height: 3.25rem;
  border-bottom: 1px solid var(--lab-line);
}
.lab-masthead .lab-label { color: var(--lab-ink); font-weight: 600; }
.lab-label, .lab-home {
  color: var(--lab-muted);
  font-size: .875rem;
  line-height: 1.45;
}
.lab-home {
  display: inline-flex;
  align-items: center;
  min-height: 2.75rem;
  text-underline-offset: .24em;
}
.lab-home:link, .lab-home:visited { color: var(--lab-muted); text-decoration-color: #726057; }
.lab-home:hover { color: var(--lab-ink); text-decoration-color: var(--lab-accent); }
.lab-home:focus-visible, .stTabs [role="tab"]:focus-visible { outline: 2px solid var(--lab-focus); outline-offset: 3px; }
#lab-workbench { padding: 1.75rem 0 1rem; }
.stApp #lab-workbench h1 {
  color: var(--lab-ink);
  font-size: 3.5rem;
  font-weight: 550;
  line-height: 1.08;
  letter-spacing: -.04em;
  padding: 0;
  margin: 0 0 .5rem;
}
#lab-workbench p, .lab-intro p, .lab-note p, .lab-empty p {
  color: var(--lab-muted);
  font-size: 1rem;
  line-height: 1.6;
  margin: 0;
}
#lab-workbench p { max-width: 65ch; }
.stTabs [role="tablist"] { gap: clamp(.65rem, 2vw, 1.75rem); }
.stTabs [role="tab"] { min-height: 2.75rem; padding: 0 .125rem; color: var(--lab-muted); }
.stTabs [role="tab"] p { font-size: 1rem; font-weight: 600; }
.stTabs [role="tab"]:hover, .stTabs [role="tab"][aria-selected="true"] { color: var(--lab-ink); }
.stTabs [role="tabpanel"] { padding-top: 1.5rem; }
.lab-intro { padding-bottom: .2rem; }
.stApp .lab-intro h2, .stApp .lab-note h3 {
  color: var(--lab-ink);
  line-height: 1.15;
  letter-spacing: -.035em;
  margin: 0 0 .5rem;
  padding: 0;
}
.stApp .lab-intro h2 { font-size: 1.75rem; font-weight: 600; }
.stApp .lab-note h3 { font-size: 1.125rem; }
.lab-note { border-top: 1px solid var(--lab-line); padding-top: .875rem; margin: .125rem 0 .375rem; }
.lab-empty {
  min-height: 15rem;
  display: grid;
  align-content: center;
  gap: .375rem;
  padding: 2rem clamp(1.25rem, 3vw, 2.5rem);
  border: 1px solid var(--lab-line);
  background: var(--lab-panel);
}
.lab-empty strong { font-size: 1.125rem; letter-spacing: -.02em; }
.lab-footer { border-top: 1px solid var(--lab-line); padding-top: 1rem; color: var(--lab-muted); font-size: .875rem; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stFileUploaderDropzone"] { border: 1px dashed #6d554a; border-radius: 4px; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height: 2.75rem; border-radius: 4px; }
.stApp [data-testid="stBaseButton-primary"] { color: var(--lab-bg); }
@media (max-width: 600px) {
  [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
  .stApp #lab-workbench h1 { font-size: 2.5rem; }
  .stTabs [role="tab"] p { font-size: .9375rem; }
}
</style>
""", unsafe_allow_html=True)


def section_intro(title: str, description: str) -> None:
    """Render the heading and description for a workbench mode."""
    st.markdown(f"<div class='lab-intro'><h2>{title}</h2><p>{description}</p></div>", unsafe_allow_html=True)


@st.cache_resource(max_entries=2)
def _cached_model(checkpoint_revision: int, profile_revision: int | None):
    return load_checkpoint(CHECKPOINT, DEVICE)


def get_model():
    profile_path = CHECKPOINT.with_name("inference_profile.json")
    try:
        profile_revision = profile_path.stat().st_mtime_ns
    except FileNotFoundError:
        profile_revision = None
    return _cached_model(CHECKPOINT.stat().st_mtime_ns, profile_revision)


def grad_cam(model, tensor, target_index, options: ScoreOptions = ScoreOptions()):
    # A private graph keeps explanations isolated from the shared live classifier.
    with torch.enable_grad():
        views = torch.cat((tensor, tensor.flip((-1,)))) if options.mirrored else tensor
        logits, features = model.forward_with_features(views.detach().requires_grad_(True))
        gradients, = torch.autograd.grad(logits[:, target_index].mean(), features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        maps = (weights * features.detach()).sum(dim=1, keepdim=True)
        if options.mirrored:
            maps = maps[:1] + maps[1:].flip((-1,))
        cam = torch.relu(maps)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def face_scores(model, tensors, options: ScoreOptions):
    if not tensors:
        return np.empty((0, len(EMOTIONS)), dtype=np.float32)

    batches = torch.cat(tensors).split(16)
    return torch.cat([expression_scores(model, batch.to(DEVICE), options).cpu() for batch in batches]).numpy()


def predict_faces(image: Image.Image, model, explain: bool, options: ScoreOptions = ScoreOptions()):
    rgb = np.asarray(image.convert("RGB"))
    detector = FaceDetector(max_side=1024)
    boxes = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    fallback = not boxes
    if fallback:
        h, w = rgb.shape[:2]
        # Helpful fallback for tightly cropped face portraits.
        boxes = [(0, 0, w, h)]
    crops = [crop_face(rgb, box, .12) for box in boxes]
    tensors = [preprocess_face(crop) for crop in crops]
    results = []
    for box, crop, tensor, probabilities in zip(boxes, crops, tensors, face_scores(model, tensors, options)):
        top = int(probabilities.argmax())
        heatmap = grad_cam(model, tensor.to(DEVICE), top, options) if explain else None
        results.append({"box": box, "probabilities": probabilities, "top": top,
                        "confidence": float(probabilities[top]), "crop": Image.fromarray(crop), "heatmap": heatmap})
    return rgb, results, fallback


def overlay_boxes(rgb, results):
    image = Image.fromarray(rgb.copy()).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, result in enumerate(results, 1):
        x, y, w, h = result["box"]
        if len(results) == 1 and x == 0 and y == 0 and w == rgb.shape[1] and h == rgb.shape[0]:
            continue
        draw.rounded_rectangle((x, y, x+w, y+h), radius=4, outline=ACCENT_RGB, width=3)
        draw.rounded_rectangle((x, max(0, y-27), x+max(w, 90), y), radius=3, fill=LABEL_RGB)
        draw.text((x+6, max(0, y-22)), f"FACE {i}  {EMOTIONS[result['top']].upper()}", fill=INK_RGB)
    return image


def probability_chart(probabilities, compact=False):
    order = np.argsort(probabilities)
    top = int(probabilities.argmax())
    colors = [ACCENT if i == top else "#72554b" for i in order]
    fig = go.Figure(go.Bar(x=probabilities[order], y=[EMOTIONS[i] for i in order], orientation="h",
                           marker_color=colors, text=[f"{probabilities[i]:.1%}" for i in order],
                           textposition="outside", cliponaxis=False))
    fig.update_layout(height=270 if not compact else 220, margin=dict(l=5,r=35,t=8,b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK,family="Segoe UI, system-ui, sans-serif"), xaxis=dict(range=[0,1],showgrid=True,
                      gridcolor=LINE,tickformat=".0%",title=None), yaxis=dict(title=None,gridcolor=LINE),
                      showlegend=False)
    return fig


class LiveExpressionProcessor(VideoProcessorBase):
    """Frame-by-frame local inference for the browser's live camera stream."""
    def __init__(self, model, options: ScoreOptions = ScoreOptions()):
        self.model = model
        self.options = options
        self.detector = FaceDetector(max_side=320)
        self.smoother = FaceScoreSmoother()

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        faces = self.detector.detect(bgr)
        crops = [cv2.cvtColor(crop_face(bgr, box, .10), cv2.COLOR_BGR2RGB) for box in faces]
        tensors = [preprocess_face(crop) for crop in crops]
        smoothed = self.smoother.update(faces, face_scores(self.model, tensors, self.options))
        for (x, y, w, h), probabilities in zip(faces, smoothed):
            top = int(probabilities.argmax())
            confidence = float(probabilities[top])
            color = ACCENT_RGB[::-1] if confidence >= .45 else (158, 192, 220)
            cv2.rectangle(bgr, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
            runner_up = int(np.argsort(probabilities)[-2])
            label = f"{EMOTIONS[top].upper()} {confidence:.0%}  ·  {EMOTIONS[runner_up]} {probabilities[runner_up]:.0%}"
            top_y = max(26, y - 10)
            cv2.rectangle(bgr, (x, top_y - 25), (x + min(w, 210), top_y + 4), LABEL_RGB[::-1], -1)
            cv2.putText(bgr, label, (x+7, top_y-5), cv2.FONT_HERSHEY_SIMPLEX, .46, INK_RGB[::-1], 1, cv2.LINE_AA)
        if len(faces) == 0:
            cv2.putText(bgr, "No face detected", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, INK_RGB[::-1], 2, cv2.LINE_AA)
        else:
            dashboard_probabilities = smoothed[0]
            height, width = bgr.shape[:2]
            panel_h = min(112, max(86, height // 4))
            top = height - panel_h
            overlay = bgr.copy()
            cv2.rectangle(overlay, (0, top), (width, height), LABEL_RGB[::-1], -1)
            cv2.addWeighted(overlay, .88, bgr, .12, 0, bgr)
            cv2.putText(bgr, "LIVE CLASS SCORES  ·  largest face  ·  not calibrated", (14, top+20),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, ACCENT_RGB[::-1], 1, cv2.LINE_AA)
            cell_w = max(1, width // len(EMOTIONS))
            for index, (name, value) in enumerate(zip(EMOTIONS, dashboard_probabilities)):
                x0 = index * cell_w + 10
                y0 = top + 38
                cv2.putText(bgr, name[:7].upper(), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, .38,
                            INK_RGB[::-1], 1, cv2.LINE_AA)
                bar_w = max(8, cell_w - 20)
                cv2.rectangle(bgr, (x0, y0+9), (x0+bar_w, y0+17), (46, 49, 58), -1)
                fill_w = int(bar_w * float(np.clip(value, 0, 1)))
                cv2.rectangle(bgr, (x0, y0+9), (x0+fill_w, y0+17),
                              ACCENT_RGB[::-1] if index == int(np.argmax(dashboard_probabilities)) else (75, 85, 114), -1)
                cv2.putText(bgr, f"{value:.0%}", (x0, y0+36), cv2.FONT_HERSHEY_SIMPLEX, .42,
                            INK_RGB[::-1], 1, cv2.LINE_AA)
        output = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        output.pts = frame.pts
        if frame.time_base is not None:
            output.time_base = frame.time_base
        return output


st.sidebar.markdown("<div class='lab-label'>PyTorch · FER-2013</div>", unsafe_allow_html=True)
st.sidebar.markdown("## Model details")
model, metadata = None, {}
if CHECKPOINT.exists():
    try:
        model, metadata = get_model()
        st.sidebar.metric("Best validation macro-F1", f"{metadata.get('best_val_macro_f1', 0):.1%}")
        st.sidebar.caption(f"Checkpoint epoch {metadata.get('epoch', '—')} · Runtime device: {DEVICE.type.upper()}")
    except Exception as exc:
        st.sidebar.error(f"Model load failed: {exc}")
else:
    st.sidebar.info("Train a checkpoint first with `python train.py --csv data/fer2013.csv`.")
st.sidebar.caption("The model predicts benchmark expression labels. It cannot determine anyone's internal emotional state.")

home_link = ""
if st.get_option("server.baseUrlPath").strip("/") == "lab":
    home_link = '<a class="lab-home" href="/" target="_top">Back to landing</a>'

st.markdown(f"""
<div class="lab-masthead">
  <span class="lab-label">Local workbench</span>
  {home_link}
</div>
<header id="lab-workbench">
  <h1>ExpressionLab</h1>
  <p>Explore how a computer vision model scores seven visible facial expressions using your camera or a still image.</p>
</header>
""", unsafe_allow_html=True)

if not CHECKPOINT.exists():
    st.warning("No trained checkpoint found. Train the model, then refresh this app.")
    st.code("python train.py --csv data/fer2013.csv --epochs 35\nstreamlit run app.py", language="powershell")
elif model is None:
    st.error("The model could not be loaded. See Model details in the sidebar.")
else:
    live_tab, image_tab, report_tab = st.tabs(["Live lab", "Image analysis", "Model report"])

    with live_tab:
        live_left, live_right = st.columns([1.25, .75], gap="large")
        with live_left:
            section_intro("Live camera", "Start your camera to see the model’s scores update as you move.")
            live_options = get_score_options(metadata, still=False)
            with st.container(border=True):
                webrtc_streamer(key="expression-live-camera", video_processor_factory=lambda: LiveExpressionProcessor(model, live_options),
                                media_stream_constraints={"video": True, "audio": False}, async_processing=True)
            st.caption("Frames are processed in this local Python session and are not recorded by ExpressionLab.")
        with live_right:
            st.markdown("""
<div class="lab-note">
  <h3>Watch the full distribution</h3>
    <p>The panel shows seven class scores for the largest face in view. Each face box shows its two highest-scoring labels. Similar scores indicate ambiguity.</p>
</div>
""", unsafe_allow_html=True)
            st.markdown("""
<div class="lab-note">
  <h3>Set up your camera</h3>
  <p>Face the camera in even light and keep your full face in frame. Hold an expression briefly to see how the smoothed scores settle.</p>
</div>
""", unsafe_allow_html=True)

    with image_tab:
        image_left, image_right = st.columns([.8, 1.2], gap="large")
        with image_left:
            section_intro("Inspect a portrait", "Select a clear face photo for local analysis.")
            uploaded = st.file_uploader("Choose JPG or PNG", type=["jpg", "jpeg", "png"], key="image-analysis-upload")
            explain = st.toggle("Generate Grad-CAM view", value=True, help="Highlights regions that influenced the selected class score; it is not a causal explanation.")
            original = None
            if uploaded is not None:
                try:
                    with Image.open(uploaded) as image:
                        if image.width * image.height > 16_000_000:
                            st.error("Image is too large. Please use one under 16 megapixels.")
                        else:
                            original = ImageOps.exif_transpose(image).convert("RGB")
                            st.image(original, caption="Input", use_container_width=True)
                except Exception:
                    st.error("Could not read that image. Choose a valid JPG or PNG.")
        with image_right:
            if original is None:
                st.markdown("""
<div class="lab-empty">
  <strong>Analysis appears here</strong>
  <p>Choose a JPG or PNG to compare the seven class scores.</p>
</div>
""", unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("Analyzing on this device…"):
                        still_options = get_score_options(metadata, still=True)
                        rgb, results, fallback = predict_faces(original, model, explain, still_options)
                    if fallback:
                        st.info("No face box detected. The image crop was still analyzed; use a centered, front-facing portrait for a more meaningful comparison.")
                    caption = "Full-image analysis" if fallback else f"{len(results)} face{'s' if len(results) != 1 else ''} detected"
                    st.image(overlay_boxes(rgb, results), caption=caption, use_container_width=True)
                    for idx, result in enumerate(results, 1):
                        emotion, score = EMOTIONS[result["top"]], result["confidence"]
                        metric_cols = st.columns(3)
                        metric_cols[0].metric(f"Face {idx} · top label", emotion)
                        metric_cols[1].metric("Top score", f"{score:.1%}")
                        margin = score - float(np.sort(result["probabilities"])[-2])
                        metric_cols[2].metric("Top-two gap", f"{margin:.1%}", help="A small gap means the model found the label ambiguous.")
                        st.plotly_chart(probability_chart(result["probabilities"]), use_container_width=True,
                                        config={"displayModeBar": False}, key=f"analysis-chart-{idx}")
                        if score < .45:
                            st.warning("Low top score. Treat this result as uncertain.")
                        if explain and result["heatmap"] is not None:
                            crop = np.asarray(result["crop"].resize((192, 192))).astype(np.float32)
                            heat = cv2.applyColorMap(np.uint8(255 * result["heatmap"]), cv2.COLORMAP_TURBO)
                            heat = cv2.cvtColor(cv2.resize(heat, (192, 192)), cv2.COLOR_BGR2RGB).astype(np.float32)
                            blended = np.clip(.58 * crop + .42 * heat, 0, 255).astype(np.uint8)
                            st.image(blended, caption="Grad-CAM saliency view · not an explanation of inner state", width=224)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

    with report_tab:
        metrics_path = ROOT / "artifacts" / "metrics.json"
        report_path = ROOT / "artifacts" / "classification_report.csv"
        curve_path = ROOT / "artifacts" / "training_curves.png"
        matrix_path = ROOT / "artifacts" / "confusion_matrix.png"
        section_intro("Benchmark results", "The report uses the held-out private test set when the dataset includes the official FER-2013 splits.")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original-view test accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
            m2.metric("Original-view test macro-F1", f"{metrics.get('test_macro_f1', 0):.1%}")
            m3.metric("Validation macro-F1", f"{metrics.get('best_validation_macro_f1', 0):.1%}")
            m4.metric("Test samples", f"{metrics.get('test_samples', 0):,}")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                if matrix_path.exists(): st.image(str(matrix_path), caption="Confusion matrix · held-out test split", use_container_width=True)
            with c2:
                if curve_path.exists(): st.image(str(curve_path), caption="Training and validation curves", use_container_width=True)
            if report_path.exists():
                st.markdown("#### Per-class performance")
                report = pd.read_csv(report_path, index_col=0)
                st.dataframe(report.loc[[name for name in EMOTIONS if name in report.index]], use_container_width=True)
            profile = metadata.get("inference_profile", {})
            test = profile.get("test", {})
            selected = test.get("selected", {})
            original = test.get("baseline", {})
            if profile.get("selected") == "mirrored" and {"accuracy", "macro_f1"} <= (selected.keys() & original.keys()):
                st.markdown("#### Still-image inference")
                st.caption("Scores average the original and mirrored views. Both methods used the same held-out test images.")
                accuracy, macro_f1 = st.columns(2)
                accuracy.metric("Still-image test accuracy", f"{selected['accuracy']:.2%}",
                                f"{100 * (selected['accuracy'] - original['accuracy']):+.2f} pp")
                macro_f1.metric("Still-image test macro-F1", f"{selected['macro_f1']:.2%}",
                                f"{100 * (selected['macro_f1'] - original['macro_f1']):+.2f} pp")
            st.download_button("Download evaluation metrics (JSON)", data=json.dumps(metrics, indent=2),
                               file_name="expressionlab_metrics.json", mime="application/json")
        else:
            st.info("Evaluation files are not present yet. Run training to completion to generate test metrics, per-class scores, and charts.")
        st.markdown("---")
        st.markdown("**Limitations:** FER-2013 labels describe dataset annotations, and its small, low-resolution images may differ from webcam conditions. Treat photo and webcam scores as model outputs. Grad-CAM highlights spatial features associated with a class score.")

st.markdown("""
<footer class="lab-footer">
  ExpressionLab is for education and research. Keep decisions about people outside the lab.
</footer>
""", unsafe_allow_html=True)
