import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow_engram.layers import Engram
from tensorflow_engram.models import engram_classifier
from tensorflow_engram.utils import HebbianTraceMonitor, plot_hebbian_trace

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(42)
tf.random.set_seed(42)

# Preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0  # Normalize to [0,1] range
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# Apply scaler for additional normalization
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape the data back to 28x28 for time steps (treating each row as a time step)
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Test data shape: {x_test.shape}")

# Create the MNIST classifier
mnist_model = engram_classifier(
    input_shape=(28, 28), # 28 time steps, each with 28 features
    hidden_dim=128,
    memory_size=64,  # Increased memory size
    num_classes=10,
    return_states=True,
    hebbian_lr=0.05  # Higher learning rate for Hebbian updates
)

# Create an early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Reduce learning rate when plateauing
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-5
)

# Compile and train the model
trace_callback = HebbianTraceMonitor(x_train[:32], log_dir="examples/out/hebbian_trace", verbose=1)

mnist_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
mnist_model.summary()
plot_model(mnist_model, to_file='examples/out/mnist_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)

# mnist_model = keras.models.load_model('examples/out/mnist_model.h5', custom_objects={'HebbianTraceMonitor': HebbianTraceMonitor, 'Engram': Engram})

history = mnist_model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[trace_callback, early_stopping, lr_scheduler],
    verbose=1
)

np.save('examples/out/mnist_history.npy', history.history)
# history = np.load('examples/out/mnist_history.npy', allow_pickle=True).item()

# Evaluate the model
print("\nRunning evaluation on test data...")
test_loss, test_acc = mnist_model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Make predictions
y_pred = mnist_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(10)], output_dict=True)
df = pd.DataFrame(report)  #.transpose()
fig, ax = plt.subplots(figsize=(len(df)+1, 8))
sns.heatmap(df.iloc[:-1, :].T, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title("Classification Report", fontsize=16)
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Classes", fontsize=12)
plt.tight_layout()
plt.savefig('examples/out/classification_report.png')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('examples/out/confusion_matrix.png')
plt.show()

# Plot the training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('examples/out/training_history.png')
plt.show()

plot_hebbian_trace(trace_callback, file_path='examples/out/hebbian_trace.png')

mnist_model.save('examples/out/mnist_model.h5')
