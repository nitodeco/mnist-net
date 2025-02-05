import streamlit as st
import torch
from torch import load
from torch.utils.data import DataLoader
from pathlib import Path
from core.model import Network, device
from core.dataset import test_data


def app():
    st.title("Model Evaluation")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Only count predictions with confidence above this threshold",
    )

    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))

    if not model_files:
        st.error("No models found")
        return

    selected_checkpoint = st.selectbox("Model", options=[f.name for f in model_files])
    model_path = next(f for f in model_files if f.name == selected_checkpoint)

    if st.button("Run Evaluation", type="primary"):
        progress = st.progress(0)
        result = st.empty()
        per_class_results = st.empty()

        model = Network().to(device)
        model.load_state_dict(load(model_path, weights_only=True))
        model.eval()

        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                confident_predictions = confidence >= confidence_threshold

                total += labels.size(0)
                correct += ((predicted == labels) & confident_predictions).sum().item()

                c = (predicted == labels).squeeze() & confident_predictions
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                progress.progress((idx + 1) / len(test_loader))

        accuracy = 100 * correct / total
        result.success(f"Overall Accuracy: {accuracy:.2f}%")

        class_results = []
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_results.append(f"Class {i}: {class_acc:.2f}%")

        chart_data = {
            "Digit": [str(i) for i in range(10)],
            "Accuracy": [
                float(result.split(": ")[1].strip("%")) for result in class_results
            ],
        }
        per_class_results.markdown("### Accuracy by Digit")
        per_class_results.bar_chart(chart_data, x="Digit", y="Accuracy", height=400)
