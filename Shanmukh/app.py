import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import ast
import os

# --- Configuration: Point to LOCAL files ---
MODEL_PATH = 'emotion_model.keras'
MUSIC_DB_PATH = 'processed_music.csv'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_music_database():
    df = pd.read_csv(MUSIC_DB_PATH)
    return df

emotion_model = load_emotion_model()
music_df = load_music_database()

st.title("AI MoodMate 🎵")
st.write("Upload an image of a face and I'll recommend some music to match your mood.")

input_option = st.radio("Choose your input method:", ("Upload a file", "Use Webcam"))
image_to_process = None

if input_option == "Upload a file":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
else:
    webcam_image = st.camera_input("Take a picture!")
    if webcam_image is not None:
        image_to_process = Image.open(webcam_image)

if image_to_process is not None:
    st.image(image_to_process, caption='Your Image', use_column_width=True)

    img_resized = image_to_process.resize((96, 96))
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')

    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    with st.spinner('Detecting your mood...'):
        predictions = emotion_model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_emotion = EMOTION_LABELS[predicted_class_index]

    st.success(f"I think you're feeling: **{predicted_emotion.capitalize()}**")

    st.subheader(f"Here are some '{predicted_emotion.capitalize()}' songs for you:")

    recommended_songs = music_df[music_df['emotion'] == predicted_emotion]

    if not recommended_songs.empty:
        for index, row in recommended_songs.sample(min(5, len(recommended_songs))).iterrows():
            try:
                artist_list = ast.literal_eval(row['artists'])
                st.write(f"- **{row['name']}** by {', '.join(artist_list)}")
            except:
                st.write(f"- **{row['name']}** by {row['artists']}")
    else:
        st.write(f"Sorry, I couldn't find any songs for '{predicted_emotion}'.")
