import firebase_admin
from firebase_admin import credentials, firestore
import json
import streamlit as st

# Initialize Firebase ONLY ONCE
if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()
