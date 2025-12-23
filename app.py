import streamlit as st
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from firebase_utils import db, bucket
import io
import json
import firebase_admin
from firebase_admin import credentials

# ---------------- Firebase Init ----------------
if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(cred)

# ---------------- Page Config ----------------
st.set_page_config(page_title="Face Registration", layout="centered")
st.title("📸 Face Registration (Mobile Friendly)")
st.success("Firebase connected successfully ✅")

# ---------------- Config ----------------
ANGLES = ["Front", "Left", "Right", "Up", "Down"]

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

model = load_model()

# ---------------- Session State ----------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "captured" not in st.session_state:
    st.session_state.captured = {}

# ---------------- UI ----------------
name = st.text_input("Student Name / ID")

st.markdown("### 📌 Capture Instructions")
st.info(
    "Capture **exactly one face** for each pose.\n\n"
    "🧍 Front → 👈 Left → 👉 Right → ⬆ Up → ⬇ Down"
)

st.markdown(f"### ✅ Captured: {len(st.session_state.captured)}/5")

# ---------------- Camera Capture (ONE at a time) ----------------
if st.session_state.step < len(ANGLES):
    angle = ANGLES[st.session_state.step]

    st.subheader(f"📷 Capture {angle} Face")
    st.caption(f"Pose {st.session_state.step + 1} of 5")

    camera_image = st.camera_input(
        f"Take {angle} photo",
        key=f"camera_{angle}"
    )

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        img_np = np.array(img)

        faces = model.get(img_np)

        if len(faces) != 1:
            st.error("❌ Exactly ONE face must be visible. Try again.")
        else:
            st.success(f"✅ {angle} face captured")

            st.session_state.captured[angle] = {
                "image": img,
                "embedding": faces[0].embedding
            }

            st.session_state.step += 1
            st.rerun()
else:
    st.success("🎯 All 5 face poses captured successfully!")

# ---------------- Register Button ----------------
if st.button("🚀 Register Face"):
    if not name:
        st.error("Please enter Student Name / ID")
        st.stop()

    if len(st.session_state.captured) != 5:
        st.error("Please capture all 5 face poses before registering")
        st.stop()

    embeddings = []

    for angle, data in st.session_state.captured.items():
        embeddings.append(data["embedding"])

        img_bytes = io.BytesIO()
        data["image"].save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        blob = bucket.blob(f"registered_faces/{name}/{angle}.jpg")
        blob.upload_from_file(img_bytes, content_type="image/jpeg")

    final_embedding = np.mean(embeddings, axis=0).tolist()

    db.collection("users").document(name).set({
        "name": name,
        "embedding": final_embedding
    })

    st.success(f"🎉 Face registered successfully for {name}")

    # Reset for next registration
    st.session_state.captured = {}
    st.session_state.step = 0
