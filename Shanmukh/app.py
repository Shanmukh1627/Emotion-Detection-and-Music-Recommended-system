# Run from your folder so relative paths work
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# --- Streamlit must be configured before any other st.* call ---
import streamlit as st
st.set_page_config(page_title="AI MoodMate 🎵", page_icon="🎵", layout="centered")

import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import ast
import gdown

# Try keras (Keras 3) first; fall back to tf.keras if needed
try:
    import keras
    _USE_KERAS = True
except Exception:
    import tensorflow as tf
    _USE_KERAS = False

# ---------- Config ----------
MODEL_PATH = BASE_DIR / "emotion_model.keras"
MUSIC_DB_PATH = BASE_DIR / "processed_music.csv"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
GDRIVE_URL = "https://drive.google.com/uc?id=17V1RvB_Wt7MbE7NHWXKzbJiXugIPNjDx"  # public link

# Download model once if not present (use print here, not st.write, before UI is ready)
if not MODEL_PATH.exists():
    print("🔽 Downloading model from Google Drive…")
    gdown.download(GDRIVE_URL, str(MODEL_PATH), quiet=False)

# ---------- Title & description ----------
st.title("🎧 AI MoodMate – Emotion Detection + Music Recommendations")
st.write("Upload a face image **or** use your **webcam**. I’ll detect the mood and suggest matching songs.")

# ---------- Load resources ----------
@st.cache_resource
def load_emotion_model():
    try:
        if _USE_KERAS:
            model = keras.models.load_model(str(MODEL_PATH))
        else:
            import tensorflow as tf  # lazy import
            model = tf.keras.models.load_model(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_music_database(csv_path: Path):
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path.name} in {csv_path.parent.name}/")
        st.stop()
    return pd.read_csv(csv_path)

emotion_model = load_emotion_model()
music_df = load_music_database(MUSIC_DB_PATH)

# ---------- Input choice ----------
input_option = st.radio("Choose your input method:", ("Upload a file", "Use Webcam"))
image_to_process = None

if input_option == "Upload a file":
    uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
else:
    webcam_image = st.camera_input("Take a picture!")
    if webcam_image is not None:
        image_to_process = Image.open(webcam_image)

# ---------- Predict + recommend ----------
if image_to_process is not None:
    st.image(image_to_process, caption="Your Image", use_container_width=True)

    img_resized = image_to_process.resize((96, 96))
    if img_resized.mode != "RGB":
        img_resized = img_resized.convert("RGB")

    img_array = img_to_array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("🧠 Detecting your mood…"):
        preds = emotion_model.predict(img_array)
        predicted_class_index = int(np.argmax(preds[0]))
        predicted_emotion = EMOTION_LABELS[predicted_class_index]

    st.success(f"I think you're feeling: **{predicted_emotion.capitalize()}**")

    st.subheader(f"🎵 Songs for a **{predicted_emotion.capitalize()}** mood")
    recs = music_df[music_df["emotion"] == predicted_emotion]

    if not recs.empty:
        recs = recs.sample(min(5, len(recs)), random_state=42)
        for _, row in recs.iterrows():
            try:
                artist_list = ast.literal_eval(str(row.get("artists", "[]")))
                artists = ", ".join(artist_list) if isinstance(artist_list, list) else str(row.get("artists", "Unknown"))
            except Exception:
                artists = str(row.get("artists", "Unknown"))
            st.write(f"- **{row.get('name', 'Untitled')}** by {artists}")
    else:
        st.info(f"Sorry, no songs found for **{predicted_emotion}**.")
else:
    st.info("⬆️ Upload an image or switch to **Use Webcam** to get started.")
