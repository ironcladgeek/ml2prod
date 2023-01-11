import os
import streamlit as st
from model import DogBreedsDetection

uploaded_file = st.file_uploader(label="Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    fp = f"./{uploaded_file.name}"
    with open(fp, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(image=fp, width=300)
    st.button("What's the breed?", key="submit_but")

    if st.session_state.submit_but:
        detector = DogBreedsDetection()
        preds = detector.predict(img_fp=fp, k=5)
        st.json(preds)
        os.remove(fp)
        
