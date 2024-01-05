import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import os

# check if gpu available
print("GPU Available: ", tf.test.is_gpu_available())
# check if eager execution enabled
print("Eager execution enabled: ", tf.executing_eagerly())
# assign to gpu if available
device = tf.device("gpu:0" if tf.test.is_gpu_available() else "cpu:0")


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add channel dimension to the images : Channels last and convert to float32
# change dtype to tf.float32

x_train = x_train[..., tf.newaxis]
x_train = tf.cast(x_train, tf.float32)
x_test = x_test[..., tf.newaxis]
x_test = tf.cast(x_test, tf.float32)

# print in a nice format the shapes of the data
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape)

# print the dtype
print("x_train dtype:", x_train.dtype, "y_train dtype:", y_train.dtype)
print("x_test dtype:", x_test.dtype, "y_test dtype:", y_test.dtype)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# Batch the datasets
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Example: Iterate through the training dataset
for img, label in train_dataset.take(3):  # Taking 3 examples for demonstration
    print("Image Shape:", img.shape, "Label:", label.numpy())


# Define a simple model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create the model
model = create_model()

# Define loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    # Reset metrics at the start of each epoch
    train_loss_metric.reset_states()
    train_accuracy_metric.reset_states()

    # Iterate through the training dataset
    for img, label in tqdm(train_dataset):
        # Perform any additional preprocessing if needed (e.g., normalization)
        img = img / 255.0  # Normalize pixel values to [0, 1]

        # Forward pass
        with tf.GradientTape() as tape:
            predictions = model(img)  # Add batch dimension


            targets = tf.squeeze(tf.convert_to_tensor([label]))

            loss = loss_fn(targets, predictions)

        # Backward pass and optimization
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update metrics
        train_loss_metric.update_state(loss)

        train_accuracy_metric.update_state(tf.squeeze(tf.convert_to_tensor([label])), predictions)

    # Print metrics at the end of each epoch
    print(f'Training - Epoch {epoch + 1}/{num_epochs}, '
          f'Loss: {train_loss_metric.result():.4f}, '
          f'Accuracy: {train_accuracy_metric.result() * 100:.2f}%')

# Test loop
    for img, label in test_dataset:
        # Perform any additional preprocessing if needed (e.g., normalization)
        img = img / 255.0  # Normalize pixel values to [0, 1]

        # Forward pass
        predictions = predictions = model(img)
        targets = tf.squeeze(tf.convert_to_tensor([label]))

        loss = loss_fn(targets, predictions)

        # Update metrics
        test_loss_metric.update_state(loss)
        test_accuracy_metric.update_state(targets, predictions)

# Print metrics for the test set
    print(f'Testing - Loss: {test_loss_metric.result():.4f}, '
          f'Accuracy: {test_accuracy_metric.result() * 100:.2f}%')


