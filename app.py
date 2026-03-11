import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Detection")

# Title
st.title("🧠 Brain Tumor Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession("model.onnx")
    return session

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("L")        # grayscale
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)   # (1,150,150)
    image = np.expand_dims(image, axis=-1)  # (1,150,150,1)
    image = np.repeat(image, 3, axis=-1)    # (1,150,150,3)

    return image

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# Show uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=350)

    input_tensor = preprocess_image(image)

    # ---------------- PREDICTION ----------------
    prediction = session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(f"❌ Tumor Detected ({confidence*100:.2f}%)")
    else:
        st.success(f"✅ No Tumor Detected ({(1-confidence)*100:.2f}%)")
            # ---------- PRECAUTIONS ----------
    st.markdown("---")
    st.subheader("📝 Precautions & Health Tips")

    if confidence > 0.5:
        st.markdown("""
        **⚠️ Tumor Detected – Suggested Precautions:**
        - Consult a **neurologist or neurosurgeon immediately**
        - Avoid stress and take **proper rest**
        - Follow **MRI / CT scan** advice from doctor
        - Do **not self-medicate**
        - Maintain a **healthy diet**
        """)
    else:
        st.markdown("""
        **✅ No Tumor Detected – General Precautions:**
        - Maintain a **healthy lifestyle**
        - Stay **hydrated**
        - Avoid excessive screen time
        - Get **regular medical checkups**
        - If symptoms persist, consult a doctor
        """)

   