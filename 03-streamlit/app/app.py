import os
import streamlit as st
from model import DogBreedsDetection

uploaded_file = st.file_uploader(label="Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # store the uploaded image into disk
    fp = f"./{uploaded_file.name}"
    with open(fp, "wb") as f:
        f.write(uploaded_file.getvalue())
    # make predictions
    detector = DogBreedsDetection()
    preds = detector.predict(img_fp=fp, k=5)
    st.image(image=fp, width=300) # show the uploaded image
    st.json(preds) # show the predictions
    os.remove(fp) # remove the uploaded image
