
import os
import streamlit as st

from src.predict import predict_image
from src.gradcam import generate_gradcam

# =====================================================
# Load CSS
# =====================================================

def load_css():

    with open("assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🩺",
    layout="wide"
)

load_css()

# =====================================================
# Output Folder
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🩺 AI Medical Assistant")

    st.markdown("---")

    st.markdown(
        """
### 📌 Project Information

This application detects **Pneumonia**
from Chest X-Ray images using
a Deep Learning model.

**Model**
- ResNet18

**Framework**
- PyTorch

**Explainability**
- Grad-CAM
"""
    )

    st.markdown("---")

    st.info(
        """
### 📋 Instructions

1. Upload a Chest X-Ray

2. Click **Predict Disease**

3. View Prediction

4. View Confidence

5. View Grad-CAM
"""
    )

    st.markdown("---")

    st.warning(
        """
⚠️ This application is for
educational purposes only.

Always consult a medical professional.
"""
    )

# =====================================================
# Main Title
# =====================================================

st.markdown(
    """
<div class="main-title">
🩺 Chest X-Ray Disease Detection
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
AI Powered Pneumonia Detection using
ResNet18 + Grad-CAM
</div>
""",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# Upload
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)




        # =====================================================
# Uploaded Image
# =====================================================

if uploaded_file is not None:

    image_path = os.path.join(
        "outputs",
        uploaded_file.name
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Chest X-Ray uploaded successfully!")

    st.markdown("## 📷 Uploaded Chest X-Ray")

    st.image(
        image_path,
        use_container_width=True
    )

    st.write("")

    # =====================================================
    # Predict Button
    # =====================================================

    if st.button(
        "🔍 Predict Disease",
        use_container_width=True
    ):

        with st.spinner("🧠 AI Model is analyzing the X-Ray..."):

            prediction, confidence = predict_image(image_path)

            gradcam_path = generate_gradcam(
                image_path,
                output_path="outputs/gradcam_result.jpg"
            )

        st.success("🎉 Analysis Completed Successfully!")

        st.divider()

        # =====================================================
        # Prediction Cards
        # =====================================================

        st.markdown("## 📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="card">

                <h2 style="text-align:center;color:#003366;">
                Prediction
                </h2>

                <div class="prediction">
                {prediction}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="card">

                <h2 style="text-align:center;color:#003366;">
                Confidence
                </h2>

                <div class="confidence">
                {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # =====================================================
        # Images
        # =====================================================

        st.markdown("## 🖼️ Original X-Ray vs Grad-CAM")

        left, right = st.columns(2)

        with left:

            st.markdown(
                """
                <div class="card">
                """,
                unsafe_allow_html=True
            )

            st.image(
                image_path,
                caption="Original Chest X-Ray",
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with right:

            st.markdown(
                """
                <div class="card">
                """,
                unsafe_allow_html=True
            )

            st.image(
                gradcam_path,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # =====================================================
# Grad-CAM Explanation
# =====================================================

st.markdown("""
<div class="card">

<h2 style="color:#003366;">
🧠 What is Grad-CAM?
</h2>

<p style="color:#222222;font-size:18px;line-height:1.8;">

Grad-CAM (Gradient-weighted Class Activation Mapping)
highlights the regions of the Chest X-Ray that influenced
the AI model's prediction.

</p>

<p style="color:#d62828;font-size:18px;font-weight:bold;">

🔴 Red / Yellow regions indicate areas that had the strongest impact.

</p>

<p style="color:#222222;font-size:18px;line-height:1.8;">

This improves transparency by showing where the AI focused
before making its prediction.

</p>

</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# About the Model
# =====================================================

st.markdown("## 🤖 About the AI Model")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
<div class="card">

### Model Details

- Architecture : **ResNet18**
- Framework : **PyTorch**
- Explainability : **Grad-CAM**
- Classes : **NORMAL / PNEUMONIA**

</div>
""",
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
<div class="card">

### Features

✅ Deep Learning Classification

✅ Chest X-Ray Analysis

✅ Confidence Score

✅ Explainable AI

✅ Medical Dashboard

</div>
""",
        unsafe_allow_html=True
    )

st.divider()

# =====================================================
# Medical Disclaimer
# =====================================================

st.warning(
    """
## ⚠ Medical Disclaimer

This application is intended **only for educational and research
purposes**.

The prediction generated by the AI model **must not be considered a
medical diagnosis**.

Always consult a qualified radiologist or healthcare professional for
clinical interpretation.
"""
)

st.divider()

# =====================================================
# Footer
# =====================================================

st.markdown(
    """
<div class="footer">

<h3>🩺 Chest X-Ray Disease Detection</h3>

Built using <b>PyTorch</b>, <b>ResNet18</b>,
<b>Streamlit</b> and <b>Grad-CAM</b>

<hr>

<p>
Artificial Intelligence for Medical Image Analysis
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>
""",
    unsafe_allow_html=True
)