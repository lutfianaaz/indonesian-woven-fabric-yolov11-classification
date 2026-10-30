
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Klasifikasi Kain Tenun Indonesia",
    page_icon="🧵",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "best.pt"

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

st.title("Klasifikasi Kain Tenun Indonesia")

st.write(
    "Aplikasi pengenalan enam jenis kain tenun Indonesia "
    "menggunakan algoritma YOLOv11."
)

uploaded_file = st.file_uploader(
    "Unggah gambar kain tenun",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("Prediksi Jenis Tenun"):

        with st.spinner("Sedang melakukan klasifikasi..."):

            result = model.predict(
                source=image,
                imgsz=224,
                verbose=False
            )[0]

            class_index = result.probs.top1
            class_name = result.names[class_index]
            confidence = float(result.probs.top1conf)

        st.success("Prediksi berhasil!")

        st.subheader("Hasil Klasifikasi")

        st.write(f"**Jenis Tenun:** {class_name}")

        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Probabilitas Setiap Kelas")

        probabilities = result.probs.data.cpu().numpy()

        for index, probability in enumerate(probabilities):

            st.write(
                f"{result.names[index]}: "
                f"{probability * 100:.2f}%"
            )
