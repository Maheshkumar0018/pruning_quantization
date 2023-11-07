import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import zipfile
import os
import tempfile
import numpy as np

class ModelCompression:
    def __init__(self):
        print("TensorFlow version:", tf.__version__)

    # load dataset
    def load_cifar10_data(self):
        data = keras.datasets.cifar10
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = data.load_data()

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')

    # build basic CNN model
    def build_base_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(32, 32, 3)))
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        return model

    def compile_model(self, model):
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, model, train_images, train_labels, epochs=8, validation_split=0.1):
        model.fit(train_images, train_labels, epochs=epochs, validation_split=validation_split)
        return model

    def evaluate_model(self, model, test_images, test_labels):
        _, accuracy = model.evaluate(test_images, test_labels)
        return accuracy

    def save_model(self, model, file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tf.keras.models.save_model(model, file_path, include_optimizer=False)

    def prune_model(self, model, end_epoch=5):
        num_iterations_per_epoch = len(self.train_images)
        end_step = num_iterations_per_epoch * end_epoch

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.25, final_sparsity=0.75, begin_step=0, end_step=end_step),
        }

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
        ]

        model_for_pruning.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy']
        )

        model_for_pruning.fit(self.train_images, self.train_labels, epochs=15, validation_split=0.1, callbacks=callbacks)
        return model_for_pruning

    def strip_and_quantize_model(self, model_for_pruning):
        ####  compressible model for TensorFlow
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        print('Saved pruned Keras model to:', pruned_keras_file)
        ####  compressible model for TFLite
        #Weights and activations in the model: Quantized to 8-bit integers (int8) for optimization.
        #Inputs and outputs of the model: Remain in their original data types, usually 32-bit float 
        # (float32), to maintain result accuracy.
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()
        _, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

        with open(quantized_and_pruned_tflite_file, 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)

        return pruned_keras_file, quantized_and_pruned_tflite_file

    def get_gzipped_model_size(self, model_path):
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(model_path)
        return os.path.getsize(zipped_file)

    def evaluate_models_on_test_images(self, base_model, pruned_keras_file, quantized_and_pruned_tflite_file):
        base_model_acc = self.evaluate_model(base_model, self.test_images, self.test_labels)
        print('*****Base model test accuracy:', base_model_acc)

        pruned_keras_model = keras.models.load_model(pruned_keras_file)
        pruned_keras_model = self.compile_model(pruned_keras_model)
        pruned_keras_acc = self.evaluate_model(pruned_keras_model, self.test_images, self.test_labels)
        print('****Pruned Keras model test accuracy:', pruned_keras_acc)

        interpreter = tf.lite.Interpreter(model_path=quantized_and_pruned_tflite_file)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        quantized_pruned_tflite_acc = 0
        for i in range(len(self.test_images)):
            input_data = self.test_images[i:i+1]
            expected_output = self.test_labels[i:i+1]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
            quantized_pruned_tflite_acc += (np.argmax(tflite_model_predictions) == expected_output[0])

        quantized_pruned_tflite_acc = float(quantized_pruned_tflite_acc) / len(self.test_images)
        print('*****Quantized and Pruned TFLite model test accuracy:', quantized_pruned_tflite_acc)



compression = ModelCompression()
compression.load_cifar10_data()
base_model = compression.build_base_model()
base_model = compression.compile_model(base_model)
compression.train_model(base_model, compression.train_images, compression.train_labels)
base_model_acc = compression.evaluate_model(base_model, compression.test_images, compression.test_labels)
compression.save_model(base_model, './baseline_model.h5')

pruned_model = compression.prune_model(base_model)
compression.evaluate_model(pruned_model, compression.test_images, compression.test_labels)

pruned_keras_file, quantized_and_pruned_tflite_file = compression.strip_and_quantize_model(pruned_model)

base_model_size = compression.get_gzipped_model_size('./baseline_model.h5')
pruned_keras_size = compression.get_gzipped_model_size(pruned_keras_file)
quantized_pruned_tflite_size = compression.get_gzipped_model_size(quantized_and_pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % base_model_size)
print("Size of gzipped pruned Keras model: %.2f bytes" % pruned_keras_size)
print("Size of gzipped pruned TFlite model: %.2f bytes" % quantized_pruned_tflite_size)

compression.evaluate_models_on_test_images(base_model, pruned_keras_file, quantized_and_pruned_tflite_file)



# Size of gzipped baseline Keras model: 965927.00 bytes
# Size of gzipped pruned Keras model: 713786.00 bytes
# Size of gzipped pruned TFlite model: 194909.00 bytes

# *****Base model test accuracy: 0.6912000179290771
# ****Pruned Keras model test accuracy: 0.6912000179290771
# *****Quantized and Pruned TFLite model test accuracy: 0.6906






#deprecation warning related to the conversion of an array with ndim > 0 to a scalar. This warning suggests that in future versions of NumPy (1.25 and later), this conversion will no longer be allowed. To address this warning and avoid potential issues in future versions of NumPy, you should explicitly extract a single element from your array before performing the division

# def evaluate_models_on_test_images(self, base_model, pruned_keras_file, quantized_and_pruned_tflite_file):
#     base_model_acc = self.evaluate_model(base_model, self.test_images, self.test_labels)
#     print('*****Base model test accuracy:', base_model_acc)

#     pruned_keras_model = keras.models.load_model(pruned_keras_file)
#     pruned_keras_model = self.compile_model(pruned_keras_model)
#     pruned_keras_acc = self.evaluate_model(pruned_keras_model, self.test_images, self.test_labels)
#     print('****Pruned Keras model test accuracy:', pruned_keras_acc)

#     interpreter = tf.lite.Interpreter(model_path=quantized_and_pruned_tflite_file)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     quantized_pruned_tflite_acc = 0
#     for i in range(len(self.test_images)):
#         input_data = self.test_images[i:i+1]
#         expected_output = self.test_labels[i:i+1]
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
#         quantized_pruned_tflite_acc += (np.argmax(tflite_model_predictions) == expected_output[0])

#     quantized_pruned_tflite_acc = float(quantized_pruned_tflite_acc) / len(self.test_images)
#     print('*****Quantized and Pruned TFLite model test accuracy:', quantized_pruned_tflite_acc)
