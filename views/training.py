import streamlit as st
import time
import os
from core.model import Network, device
from core.train import train
from torch import save
import shutil


def app():
    st.title("Training")

    if st.button("Clear Models", type="secondary"):
        try:
            shutil.rmtree("models")
            os.makedirs("models", exist_ok=True)
            st.success("Models cleared")
        except Exception as e:
            st.error(f"Failed to clear models: {str(e)}")

    params = {
        "epochs": st.slider("Epochs", 1, 16, 8),
        "batch_size": st.slider("Batch Size", 8, 256, 32, step=8),
        "learning_rate": st.number_input(
            "Learning Rate", 0.0001, 0.01, 0.0015, format="%.4f"
        ),
    }

    if "training_active" not in st.session_state:
        st.session_state.training_active = False

    progress_bar = st.empty()
    status_text = st.empty()

    col1, col2 = st.columns(2)
    start = col1.button("Start Training", type="primary")
    stop = col2.button("Stop Training", type="secondary")

    if start:
        st.session_state.training_active = True

        def update_progress(
            progress: float, loss: float, eta: float, epoch: int
        ) -> None:
            if stop:
                st.session_state.training_active = False
                raise InterruptedError("Training stopped by user")

            progress_bar.progress(progress)
            status_text.text(
                f"Epoch: {epoch+1} | Loss: {loss:.4f} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
            )

        try:
            train(callback=update_progress, **params)
            st.success("Training complete")
        except InterruptedError:
            st.warning("Training stopped by user")
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            model = Network().to(device)
            save(model.state_dict(), f"models/mnist_model_interrupted_{timestamp}.pth")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            st.session_state.training_active = False


def cleanup_training_ui():
    st.session_state.training_active = False
