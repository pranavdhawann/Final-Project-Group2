import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

@st.cache_resource
def load_model():
    #model = torch.load()
    return model

def main():
    st.title("Bacterial Flagellar Motor Detection")
    st.write("Upload a tomogram image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        img_array = np.array(image)

        st.image(image, caption="Uploaded Tomogram", use_column_width=True, clamp=True)
        st.write("Running detection...")

        model_input = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        model = load_model()

        results = model(model_input)

        detected_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)  # For annotation purposes

        detections = results.pandas().xyxy[0]

        for _, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            confidence = det['confidence']
            label = f"{det['name']} {confidence:.2f}"

            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            cv2.putText(detected_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        detected_img_pil = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY))

        st.image(detected_img_pil, caption="Detection Results", use_column_width=True, clamp=True)
        st.write("Detection Details:")
        st.dataframe(detections)

if __name__ == "__main__":
    main()