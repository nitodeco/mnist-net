# MNIST-Net

A basic implementation of a linear neural network trained on the [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database). I used this as an excercise to apply my theoretical knowledge about neural networks and machine learning and experiment with realistic data. The network is defined, trained and evaluated using [PyTorch](https://pytorch.org/).

# Usage

1. Create and activate your python environment
2. Install dependencies

```bash
pip3 install -r requirements.txt
```

3. Download the dataset

```bash
python3 ./common/dataset.py
```

4. Train and evaluate the model

```bash
python3 ./main.py
```
