import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load model
# -------------------------------
model = load_model("brain_tumor_multiclass.keras")

# Class labels (ORDER MUST MATCH TRAINING)
classes = ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]

# Medical info (rule-based, informational)
info = {
    "Glioma": {
        "region": "Frontal / Temporal Lobe",
        "symptoms": [
            "Persistent headaches",
            "Seizures",
            "Memory loss",
            "Personality changes"
        ],
        "treatment": [
            "Surgery",
            "Radiation therapy",
            "Chemotherapy"
        ]
    },
    "Meningioma": {
        "region": "Brain covering (Meninges)",
        "symptoms": [
            "Headaches",
            "Vision problems",
            "Seizures"
        ],
        "treatment": [
            "Observation",
            "Surgery",
            "Radiation therapy"
        ]
    },
    "Pituitary Tumor": {
        "region": "Pituitary gland",
        "symptoms": [
            "Hormonal imbalance",
            "Vision problems",
            "Weight changes"
        ],
        "treatment": [
            "Medication",
            "Surgery",
            "Radiation therapy"
        ]
    },
    "No Tumor": {
        "region": "—",
        "symptoms": ["No abnormal symptoms detected"],
        "treatment": ["No treatment required"]
    }
}

# -------------------------------
# App UI
# -------------------------------
st.title("🧠 Brain Tumor Detection & Classification")
st.write(
    "Upload a **brain MRI image** to detect whether a tumor is present "
    "and identify its **type**."
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 128, 128, 3)

    # Prediction
    prediction = model.predict(img_reshaped)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_class = classes[predicted_class_index]

    st.markdown("---")

    # Result
    if predicted_class != "No Tumor":
        st.error(f"🛑 **Tumor Detected**: {predicted_class}")
    else:
        st.success("✅ **No Tumor Detected**")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.markdown("### 🧬 Medical Insights (Informational)")
    st.write(f"**Likely Affected Region:** {info[predicted_class]['region']}")

    st.markdown("**Possible Symptoms:**")
    for symptom in info[predicted_class]["symptoms"]:
        st.write(f"- {symptom}")

    st.markdown("**Common Treatment Options:**")
    for treatment in info[predicted_class]["treatment"]:
        st.write(f"- {treatment}")

    st.markdown("---")
    st.caption(
        "⚠️ *Disclaimer: This system is for educational purposes only and "
        "should not be used as a medical diagnostic tool.*"
    )
