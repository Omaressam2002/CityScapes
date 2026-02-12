import streamlit as st
from PIL import Image
import os
import requests 

st.title("Drag & Drop Image Upload")

uploaded_file = st.file_uploader(
    "Drop an image here",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # File name (title)
    filename = uploaded_file.name
    st.write("📄 File name:", filename)

    # Read & display image
    image = Image.open(uploaded_file)
    # should i create an end point ot the model here that sends the image path and returns the image predicted
    st.image(image, caption=filename)

    # Save locally if you want a path
    save_dir = "/data/results"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    url = "http://backend:8000/health"  # 🔥 NEW
    response = requests.get(              # 🔥 NEW
        url
    )
    
    st.success(f"Device Used : {response.json()['device']}")
    url = "http://backend:8000/predict"  # 🔥 NEW
    response = requests.post(              # 🔥 NEW
        url,
        json={"image_path": file_path}
    )

    if response.status_code == 200:        # 🔥 NEW
        mask_path = response.json()["mask_path"]
        st.success("Prediction completed!")
        st.success(mask_path)
        st.write("🧠 Mask saved at:", mask_path)

        # Optional: display returned mask
        mask_image = Image.open(mask_path)  # 🔥 NEW
        st.image(mask_image, caption="Predicted Mask")  # 🔥 NEW
    else:
        st.error("Prediction failed!")     # 🔥 NEW

    st.write("💾 Saved to:", file_path)
