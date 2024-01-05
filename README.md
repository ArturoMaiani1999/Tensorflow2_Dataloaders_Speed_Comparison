# TensorflowBasics

This tutorial aims at defining the best practices in using tensorflow 2x. 

The TFRecords script has been taken from https://www.kaggle.com/code/mmenendezg/custom-cnn-for-mnist-using-tfrecords/notebook

## Installation

Create a python 3.10 environment and install the requirements.txt file.

## DataLoaders



In TensorFlow, creating a data loader typically involves using the `tf.data.Dataset` API, which provides a high-performance and flexible way to load and preprocess data for training machine learning models. Here are some common ways to create a data loader using `tf.data.Dataset` or `tf.data`:

1. **From Numpy arrays:**
   ```python
   import tensorflow as tf
   # Assume you have numpy arrays for features and labels
   features = np.array(...)  # Your feature data
   labels = np.array(...)    # Your label data
   # Create a tf.data.Dataset from numpy arrays
   dataset = tf.data.Dataset.from_tensor_slices((features, labels))
   ```
try it using the script `Dataloader_from_slices.py`
2. **From TensorFlow Records (TFRecord):**
   ```python

   # Assuming you have TFRecord files
   file_pattern = '/path/to/tfrecord/files/*.tfrecord'
   # Create a tf.data.Dataset from TFRecord files
   dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
   ```
try it using the script `Dataloaders_TFRecords.py`

3. **From custom Python generators:**
   ```python
   dataset = tf.data.Dataset.from_generator(
       custom_generator,
       output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32),
                        tf.TensorSpec(shape=(), dtype=tf.int32))
   )
   ```
try it using the script `Dataloader_generator.py`

The results are obtained by training on the whole 60K MNIST images for 5 epochs with the same network.
# draw a table
| Dataloader | Time |
|------------|------|
| FromSlices | 75s  |
| TFRecord   | 60s  |
| Generator  | 75s  |