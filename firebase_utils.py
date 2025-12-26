import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase (run only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(
        cred,
        {
            "storageBucket": "face-registration-app.firebasestorage.app"
        }
    )

# Firestore and Storage clients
db = firestore.client()
bucket = storage.bucket()
