# tensorflow-engram

[![PyPI Downloads](https://img.shields.io/pypi/dm/tensorflow-engram.svg?label=PyPI%20downloads)](
https://pypi.org/project/tensorflow-engram/)
[![Conda Downloads](https://img.shields.io/conda/dn/danielathome19/tensorflow-engram.svg?label=Conda%20downloads)](
https://anaconda.org/danielathome19/tensorflow-engram)
[![CI/CT/CD](https://github.com/danielathome19/Engram-Neural-Network/actions/workflows/package_upload.yml/badge.svg)](https://github.com/danielathome19/Engram-Neural-Network/actions/workflows/package_upload.yml)
[![License](https://img.shields.io/badge/license-BSD_3_Clause-blue)](./LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.48550/arxiv.TBD.svg)](https://doi.org/10.48550/arxiv.TBD)



### Hebbian Memory-Augmented Recurrent Networks: Engram Neural Networks (ENNs)
**Biologically-inspired memory for TensorFlow/Keras.**

_Add Hebbian/engram learning to your neural networks with just a few lines of code._

---

## Overview

**tensorflow-engram** provides Keras layers, models, and utilities for building neural networks with biologically-inspired memory mechanisms, including Hebbian plasticity, engram-like trace formation, attention, and sparse memory recall. 
This enables powerful sequence modeling, few-shot learning, continual learning, and analysis of memory traces within modern deep learning pipelines.

- **Seamless TensorFlow/Keras integration**
- **Engram layers:** RNN cells and wrappers with memory banks, plastic synapses, and sparsity
- **Hebbian learning:** Fast local updates + gradient learning
- **Attention and sparsity:** Focuses on the most relevant memories
- **Trace monitoring:** Visualize engram and memory trace evolution
- **Ready-to-use models** for classification and regression


Tensorflow-Engram is currently in development and may not yet be ready for production use. We are actively seeking contributors
to help us improve the package and expand its capabilities. If you are interested in contributing, please see our
[contributing guide](CONTRIBUTING.md).

### TODO:
* Add unit tests
* Generate doc pages with Sphinx
* Possibly rename repo?

---

## Installation

```bash
pip install tensorflow-engram
```

Or install using `conda`:

```bash
conda install -c danielathome19 tensorflow-engram
```

---

## Requirements:

* Python 3.12+
* TensorFlow 2.19+
* Keras 3.10+
* numpy, seaborn, matplotlib, pandas (for utilities and plotting)

---

## Quickstart

Example: MNIST Classification with Engram Memory

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow_engram.models import EngramClassifier
from tensorflow_engram.utils import HebbianTraceMonitor, plot_hebbian_trace

# Prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
x_train = x_train.reshape(-1, 28, 28)
x_test  = x_test.reshape(-1, 28, 28)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

# Build model
model = EngramClassifier(
    input_shape=(28, 28),
    num_classes=10,
    hidden_dim=128,
    memory_size=64,
    return_states=True,
    hebbian_lr=0.05,
)

# Monitor Hebbian trace during training
trace_callback = HebbianTraceMonitor(x_train[:32], log_dir=None)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[trace_callback]
)

# Visualize trace evolution
plot_hebbian_trace(trace_callback)
```

---

## Features

* __EngramCell:__ Biologically-inspired RNN cell with memory banks and Hebbian plasticity.
* __EngramNetwork:__ High-level Keras Model for sequence modeling.
* __Attention Layer:__ Optional attention mechanism for sequence summarization.
* __Trace Monitoring:__ Inspect and visualize memory trace evolution with built-in callbacks and plotting utilities.

---

## API Highlights

### Layers

* `EngramCell`:
Biologically-inspired RNN cell with memory banks, Hebbian trace, and sparsity regularization.

* `Engram`:
Wrapper for Keras models/networks using EngramCell.

* `EngramAttentionLayer`:
Optional attention over sequence outputs.

### Models

* `EngramNetwork`:
General-purpose sequence model with configurable memory and plasticity.

* `EngramClassifier`:
Factory function for classification tasks.

* `EngramRegressor`:
Factory for regression tasks.

### Utilities

* `HebbianTraceMonitor`:
Keras callback for logging and visualizing Hebbian traces.

* `plot_hebbian_trace`:
Quick plotting of trace evolution and statistics.

---

## How It Works

* __Memory Bank:__
Persistent, learnable memory vectors (engrams), updated via gradient descent.

* __Hebbian Trace:__
Rapidly updated, plastic component reflecting short-term memory, updated via local Hebbian learning.

* __Attention/Recall & Sparsity:__
Memories are retrieved by attention (cosine similarity + softmax), but with sparsity constraints so only a few are activated per input—mimicking efficient biological memory recall.

* __Trace Visualization:__
Built-in tools to monitor and understand the dynamics of memory during training.

---

## Advanced Usage

You can customize the cell and models for your own tasks:

```python
from tensorflow_engram.layers import EngramCell
from tensorflow.keras.layers import RNN, Input
from tensorflow.keras.models import Model

cell = EngramCell(hidden_dim=64, memory_size=32)
inputs = Input(shape=(None, 16))
rnn_layer = RNN(cell, return_sequences=True)
outputs = rnn_layer(inputs)
model = Model(inputs, outputs)
```

---

## License

Tensorflow-Engram is licensed under the BSD-3 License. See the [LICENSE](LICENSE.md) file for more information.

<!-- Project development began May 1st, 2025. -->

---

## Citation

If you use this code for your research, please cite this project as:

```bibtex
@software{Szelogowski_tensorflow_engram_2025,
 author = {Szelogowski, Daniel},
 doi = {10.48550/arxiv.TBD},
 license = {BSD-3-Clause},
 month = {jul},
 title = {{tensorflow-engram: A Python package for Engram Neural Networks, adding biologically-inspired Hebbian memory and engram layers to TensorFlow/Keras models, supporting memory traces, plasticity, attention, and sparsity for neural sequence learning.}},
 url = {https://github.com/danielathome19/Engram-Neural-Network},
 version = {0.1.0},
 year = {2025}
}
```

or as the corresponding research paper:

```bibtex
@misc{Szelogowski_Simulation_of_Neural_Responses_Using_OI_2024,
 author = {Szelogowski, Daniel},
 doi = {10.48550/arxiv.TBD},
 month = {jul},
 title = {{Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning}},
 url = {https://github.com/danielathome19/Engram-Neural-Network},
 year = {2025}
}
```