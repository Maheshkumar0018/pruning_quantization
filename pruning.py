import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import zipfile
import numpy as np
import os

print("tensorflow version:", tf.__version__) #tensorflow version: 2.14.0 / Python 3.11.4


#Cifar10 Dataset
data = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#normalizing images
train_images = train_images/255.0
test_images = test_images/255.0
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Build a base model for Cifar-10 Dataset
# CNN model
class CustomCNNModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
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

    def compile_model(self):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, train_images, train_labels, epochs=8, validation_split=0.1):
        self.model.fit(train_images, train_labels, epochs=epochs, validation_split=validation_split)

    def summary(self):
        self.model.summary()

    def model_evaluate(self):
        self.model.evaluate(test_images, test_labels)

    def model_save(self, model_filename='base_model.zip'):
            self.model.save('custom_cnn_model.h5')
            with zipfile.ZipFile(model_filename, 'w') as model_zip:
                model_zip.write('custom_cnn_model.h5', 'custom_cnn_model.h5')
            print(f'Model saved in {model_filename}')


custom_cnn = CustomCNNModel()
custom_cnn.compile_model()
custom_cnn.summary()
custom_cnn.train_model(train_images, train_labels)
custom_cnn.model_evaluate()
custom_cnn.model_save()


###/// Using Tensorflow model optimization Kit for model pruning and compressing ////
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after after 5 epochs.
end_epoch = 5
num_iterations_per_epoch = len(train_images)
end_step =  num_iterations_per_epoch * end_epoch

# Define parameters for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                               final_sparsity=0.75,
                                                               begin_step=0,
                                                               end_step=end_step),
      #'pruning_policy': tfmot.sparsity.keras.PruneForLatencyOnXNNPack()
}

# load base model
# other wise use pretrained model
loaded_model = keras.models.load_model('./custom_cnn_model.h5')
model_for_pruning = prune_low_magnitude(loaded_model, **pruning_params)

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

model_for_pruning.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model_for_pruning.fit(train_images, train_labels, epochs=15, validation_split=0.1,callbacks=callbacks)
model_for_pruning.summary()
_, pruned_model_accuracy = model_for_pruning.evaluate(test_images,test_labels, verbose=0)
print('Pruned model test accuracy:', pruned_model_accuracy)

model_for_pruning.save('pruned_model.h5')
with zipfile.ZipFile('pruned_model.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('pruned_model.h5')

# before pruning base model loss: 0.9464 - accuracy: 0.7028
# Pruned model test accuracy: 0.6970999836921692



# Check the size of the original model's zip file
original_model_size = os.path.getsize('base_model.zip')
print(f"Size of the original model's zip file: {original_model_size} bytes")

# Check the size of the pruned model's zip file
pruned_model_size = os.path.getsize('pruned_model.zip')
print(f"Size of the pruned model's zip file: {pruned_model_size} bytes")

# Calculate the size reduction
size_reduction = original_model_size - pruned_model_size
print(f"Size reduction: {size_reduction} bytes")

# Convert sizes to megabytes for readability
original_model_size_MB = original_model_size / (1024 * 1024)
pruned_model_size_MB = pruned_model_size / (1024 * 1024)
size_reduction_MB = size_reduction / (1024 * 1024)

print(f"Size of the original model's zip file: {original_model_size_MB:.2f} MB")
print(f"Size of the pruned model's zip file: {pruned_model_size_MB:.2f} MB")
print(f"Size reduction: {size_reduction_MB:.2f} MB")

#Size of the original model's zip file: 3162368 bytes
#Size of the pruned model's zip file: 2062613 bytes
#Size reduction: 1099755 bytes
#Size of the original model's zip file: 3.02 MB
#Size of the pruned model's zip file: 1.97 MB
#Size reduction: 1.05 MB