import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image
from core.model import Network, device
from torch import load
from pathlib import Path


def preprocess_image(image_data):
    image = Image.fromarray(image_data).convert("L")
    image = image.resize((28, 28))
    image_tensor = torch.FloatTensor(np.array(image))
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0) / 255.0
    return image_tensor


def load_model(model_path):
    try:
        model = Network().to(device)
        checkpoint = load(model_path, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def app():
    st.title("Testing")

    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))

    if not model_files:
        st.error("No model files found")
        return

    selected_checkpoint = st.selectbox("Model", options=[f.name for f in model_files])
    model_path = next(f for f in model_files if f.name == selected_checkpoint)

    model = load_model(model_path)
    if model is None:
        return

    column_left, column_right = st.columns([1, 1])

    with column_left:
        canvas_result = st_canvas(
            stroke_width=16,
            stroke_color="#fff",
            background_color="#000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with column_right:
        if (
            canvas_result.image_data is not None
            and np.max(canvas_result.image_data) > 0
        ):
            image_tensor = preprocess_image(canvas_result.image_data)

            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                output = model(image_tensor)
                prediction = torch.argmax(output).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0]

                st.markdown(
                    f"### Prediction: **{prediction}** ({confidence[prediction]:.1%})"
                )

                chart_data = {
                    "Digit": [str(i) for i in range(10)],
                    "Confidence": [conf.item() * 100 for conf in confidence],
                }

                st.bar_chart(
                    chart_data,
                    x="Digit",
                    y="Confidence",
                    height=400,
                )
