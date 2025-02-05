# MNIST-Net

Implementation of a neural network trained on the [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database). I used this as an excercise to apply my theoretical knowledge about neural networks and machine learning and experiment with realistic data. This project is built with [PyTorch](https://pytorch.org/).

# Usage

1. Create and activate your python environment
2. Install dependencies

```bash
pip3 install -r requirements.txt
```

3. Download the dataset

```bash
python3 ./core/dataset.py
```

4. Start the app

```bash
streamlit run app.py
```

Now you can visit the app at [http://localhost:8501](http://localhost:8501) and start training, evaluating and testing the model.
