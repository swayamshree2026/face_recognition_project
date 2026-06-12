import streamlit as st
import cv2
import joblib
import numpy as np
from PIL import Image

model = joblib.load("face_model.pkl")
encoder = joblib.load("label_encoder.pkl")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Face Recognition System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.flatten().reshape(1, -1)

        prediction = model.predict(face)
        name = encoder.inverse_transform(prediction)[0]

        cv2.rectangle(
            image_np,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image_np,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(image_np)