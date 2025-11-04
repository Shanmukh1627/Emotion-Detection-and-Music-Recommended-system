from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import streamlit as st
st.set_page_config(page_title="AI MoodMate 🎵", page_icon="🎵", layout="centered")
st.markdown(
    """
    <style>
      .app-title {font-size: 2.4rem; font-weight: 800; letter-spacing:.2px;}
      .sub {color:#616995; margin-bottom:.6rem}
      .pill {display:inline-block;padding:4px 10px;border-radius:999px;background:#f2f4ff;color:#4a4fad;font-size:.8rem;margin-right:6px}
      .panel {padding:18px;border-radius:16px;background:#ffffff;border:1px solid #eef0f6;box-shadow:0 2px 10px rgba(0,0,0,.04)}
      .footer {color:#8a8ea8; font-size:.9rem; padding-top:18px}
      .rec-card {padding:14px 16px; border-radius:14px; background:#fff; box-shadow: 0 1px 8px rgba(0,0,0,.05); margin-bottom:10px;}
      .stProgress > div > div > div { background: linear-gradient(90deg,#6C63FF,#49c5b6) }
      .stRadio > div { gap: 10px }
      .st-emotion-cache-dvne4q {padding-top: 0 !important}
      a { text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-title">🎧 AI MoodMate</div>
    <div class="sub">Emotion detection from face + mood-matched music picks.</div>
    <div class="pill">Streamlit</div><div class="pill">TensorFlow/Keras</div><div class="pill">CV + Recsys</div>
    """,
    unsafe_allow_html=True,
)

import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import ast
import gdown

try:
    import keras
    _USE_KERAS = True
except Exception:
    import tensorflow as tf
    _USE_KERAS = False

MODEL_PATH = BASE_DIR / "emotion_model.keras"
MUSIC_DB_PATH = BASE_DIR / "processed_music.csv"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
GDRIVE_URL = "https://drive.google.com/uc?id=17V1RvB_Wt7MbE7NHWXKzbJiXugIPNjDx"

if not MODEL_PATH.exists():
    gdown.download(GDRIVE_URL, str(MODEL_PATH), quiet=False)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/encharm/Font-Awesome-SVG-PNG/master/black/png/64/headphones.png", width=64)
    st.markdown("### How to use")
    st.write("• Upload a face photo **or** use the **Webcam**.\n• Wait for prediction.\n• Get mood-matched songs.")
    st.markdown("### About")
    st.write("Built with TensorFlow + Streamlit.")
    st.caption("Tip: better light → better predictions.")
    st.divider()
    if st.button("🔁 Reset app"):
        st.experimental_rerun()

@st.cache_resource
def load_emotion_model():
    if _USE_KERAS:
        return keras.models.load_model(str(MODEL_PATH))
    else:
        import tensorflow as tf
        return tf.keras.models.load_model(str(MODEL_PATH))

@st.cache_data
def load_music_database(csv_path: Path):
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path.name}")
        st.stop()
    return pd.read_csv(csv_path)

emotion_model = load_emotion_model()
music_df = load_music_database(MUSIC_DB_PATH)

tab1, tab2 = st.tabs(["📁 Upload", "📷 Webcam"])
image_to_process = None
with tab1:
    uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)
with tab2:
    webcam_image = st.camera_input("Take a picture!")
if webcam_image is not None:
    image_to_process = Image.open(webcam_image)
    image_to_process = image_to_process.transpose(Image.FLIP_LEFT_RIGHT)  


if image_to_process is not None:
    try:
        st.image(image_to_process, caption="Your Image", use_container_width=True)
    except TypeError:
        st.image(image_to_process, caption="Your Image", use_column_width=True)

    img_resized = image_to_process.resize((96, 96))
    if img_resized.mode != "RGB":
        img_resized = img_resized.convert("RGB")

    arr = img_to_array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0) / 255.0

    with st.spinner("Detecting your mood…"):
        preds = emotion_model.predict(arr)
        probs = preds[0].astype(float)
        idx = int(np.argmax(probs))
        predicted_emotion = EMOTION_LABELS[idx]
        confidence = float(probs[idx])

    c1, c2 = st.columns([1,1])
    with c1:
        st.success(f"**Mood:** {predicted_emotion.capitalize()}")
    with c2:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    st.progress(min(1.0, max(0.0, confidence)))

    st.subheader(f"🎵 Songs for a **{predicted_emotion.capitalize()}** mood")
    recs = music_df[music_df["emotion"] == predicted_emotion]
    if not recs.empty:
        recs = recs.sample(min(5, len(recs)), random_state=42)
        for _, row in recs.iterrows():
            name = str(row.get("name", "Untitled"))
            try:
                artists_raw = row.get("artists", "[]")
                artist_list = ast.literal_eval(str(artists_raw)) if isinstance(artists_raw, str) else artists_raw
                artists = ", ".join(artist_list) if isinstance(artist_list, list) else str(artists_raw)
            except Exception:
                artists = str(row.get("artists", "Unknown"))
            url = str(row.get("url", "")).strip()
            label = f"**{name}** · {artists}"
            if url and url.startswith("http"):
                st.markdown(f'<div class="rec-card"><a href="{url}" target="_blank">{label}</a></div>', unsafe_allow_html=True)
            else:
                q = f"{name} {artists}".replace(" ", "+")
                yt = f"https://www.youtube.com/results?search_query={q}"
                st.markdown(f'<div class="rec-card"><a href="{yt}" target="_blank">{label}</a></div>', unsafe_allow_html=True)
    else:
        st.info(f"Sorry, no songs found for **{predicted_emotion}**.")
else:
    st.info("Upload an image or switch to **Use Webcam** to get started.")

st.markdown('<div class="footer">© 2025 MoodMate · Built by Shanmukh</div>', unsafe_allow_html=True)
