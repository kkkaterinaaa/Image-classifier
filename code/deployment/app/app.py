import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict"):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        with st.spinner('Making prediction...'):
            response = requests.post("http://fastapi-api:8000/predict/", files={"file": img_bytes})
        if response.status_code == 200:
            st.success(f"Predicted class: {response.json()['predicted_class']}")
        else:
            st.error("Error: Could not get prediction.")
