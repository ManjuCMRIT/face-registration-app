import streamlit as st
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from firebase_utils import db, bucket
import io

import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials

if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(cred)

st.success("Firebase connected successfully ✅")


st.set_page_config(page_title="Face Registration", layout="centered")
st.title("📸 Face Registration")

ANGLES = ["Front", "Left", "Right", "Up", "Down"]

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

model = load_model()

name = st.text_input("Student Name / ID")
files = st.file_uploader(
    "Upload exactly 5 images (Front, Left, Right, Up, Down)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.button("Register Face"):
    if not name or not files or len(files) != 5:
        st.error("Please enter name and upload exactly 5 images.")
        st.stop()

    embeddings = []

    for i, file in enumerate(files):
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        faces = model.get(img_np)
        if len(faces) != 1:
            st.error(f"{ANGLES[i]} image must contain exactly ONE face.")
            st.stop()

        face = faces[0]
        embeddings.append(face.embedding)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        blob = bucket.blob(f"registered_faces/{name}/{ANGLES[i]}.jpg")
        blob.upload_from_file(img_bytes, content_type="image/jpeg")

    final_embedding = np.mean(embeddings, axis=0).tolist()

    db.collection("users").document(name).set({
        "name": name,
        "embedding": final_embedding
    })

    st.success(f"✅ Face registered successfully for {name}")
