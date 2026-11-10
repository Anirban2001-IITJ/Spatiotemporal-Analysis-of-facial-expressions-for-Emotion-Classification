import os
import tempfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Emotion Classification", page_icon="🎭", layout="wide")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_URL = (
    "https://raw.githubusercontent.com/oarriaga/face_classification/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
)
MODEL_PATH = "fer2013_mini_XCEPTION.102-0.66.hdf5"
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    from tensorflow.keras.models import load_model

    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return load_model(MODEL_PATH, compile=False)


def preprocess(bgr):
    """Face detect -> tight crop -> CLAHE -> 64x64 (same as notebook)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )
    box = None
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        box = (x, y, w, h)
        dw, dh = int(w * 0.1), int(h * 0.1)
        roi = gray[y + dh : y + h - dh, x + dw : x + w - dw]
        roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    else:
        roi = cv2.equalizeHist(gray)
    roi = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
    return roi[None, :, :, None], box


def predict(model, bgr):
    x, box = preprocess(bgr)
    probs = model.predict(x, verbose=0)[0]
    return probs, box


def annotate(bgr, box, label):
    out = bgr.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def analyse_video(model, path, progress):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Notebook: 1st and 15th frame of each second (at 30 fps) -> offsets 0 and fps//2
    offsets = {0, fps // 2}
    rows, thumbs, idx = [], {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % fps in offsets:
            probs, box = predict(model, frame)
            label = EMOTIONS[int(np.argmax(probs))]
            rows.append({"Second": idx // fps, "Frame": idx, "Predicted": label,
                         "Confidence": float(probs.max()),
                         **{e: float(p) for e, p in zip(EMOTIONS, probs)}})
            thumbs[idx] = annotate(frame, box, label)
        idx += 1
        if total:
            progress.progress(min(idx / total, 1.0))
    cap.release()
    return pd.DataFrame(rows), thumbs, fps, total


st.title("🎭 Spatiotemporal Emotion Classification")
st.caption("Pre-trained FER2013 mini-XCEPTION · Haar face detection · CLAHE preprocessing")

model = get_model()
tab_video, tab_image = st.tabs(["🎬 Video analysis", "📷 Image / webcam"])

# ---------------- Video ----------------
with tab_video:
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    expected = st.selectbox(
        "Expected emotion in clip (optional, for accuracy)", ["—"] + EMOTIONS
    )
    if up and st.button("Analyse", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as t:
            t.write(up.read())
        bar = st.progress(0.0, text="Processing frames…")
        try:
            df, thumbs, fps, total = analyse_video(model, t.name, bar)
        finally:
            os.unlink(t.name)
        bar.empty()
        if df.empty:
            st.error("No frames processed.")
            st.stop()
        st.session_state["res"] = (df, thumbs, fps, total, expected)

    if "res" in st.session_state:
        df, thumbs, fps, total, expected = st.session_state["res"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FPS", fps)
        c2.metric("Total frames", total)
        c3.metric("Duration (s)", f"{total / fps:.1f}")
        c4.metric("Dominant emotion", df["Predicted"].mode()[0])
        if expected != "—":
            acc = accuracy_score([expected] * len(df), df["Predicted"])
            st.metric(f"Accuracy vs “{expected}”", f"{acc:.1%}")

        timeline = df.groupby("Second")[EMOTIONS].mean()
        st.subheader("Emotion timeline heatmap")
        fig, ax = plt.subplots(figsize=(max(8, len(timeline) * 0.6), 4))
        sns.heatmap(timeline.T, cmap="magma", annot=len(timeline) <= 30,
                    fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_xlabel("Time (seconds)")
        st.pyplot(fig)

        st.subheader("Probability over time")
        st.line_chart(timeline)

        st.subheader("Predicted frames")
        cols = st.columns(5)
        for i, (fi, img) in enumerate(list(thumbs.items())[:20]):
            cols[i % 5].image(img, caption=f"f{fi}: {df.loc[df.Frame == fi, 'Predicted'].iloc[0]}")

        st.subheader("Data")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "emotions.csv", "text/csv")

# ---------------- Image / webcam ----------------
with tab_image:
    src = st.radio("Source", ["Webcam", "Upload"], horizontal=True)
    img_file = (st.camera_input("Take a photo") if src == "Webcam"
                else st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]))
    if img_file:
        bgr = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        probs, box = predict(model, bgr)
        label = EMOTIONS[int(np.argmax(probs))]
        a, b = st.columns(2)
        a.image(annotate(bgr, box, label), caption="Detected face" if box is not None
                else "No face found – full frame used")
        b.subheader(f"{label} ({probs.max():.1%})")
        b.bar_chart(pd.Series(probs, index=EMOTIONS))
