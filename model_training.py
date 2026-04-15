import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define image size and batch size
IMG_SIZE = 64  # You can change this based on your dataset
BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = "skin_disease_model.h5"
# 1. Data Preprocessing
# Assuming the images are stored in directories corresponding to class labels, e.g., 'train' and 'test' directories
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory('skin-disease-datasaet/train_set',  
                                               target_size=(IMG_SIZE, IMG_SIZE), 
                                               batch_size=BATCH_SIZE, 
                                               class_mode='categorical')
test_data = test_datagen.flow_from_directory('skin-disease-datasaet/test_set', 
                                             target_size=(IMG_SIZE, IMG_SIZE), 
                                             batch_size=BATCH_SIZE, 
                                             class_mode='categorical')
# 2. Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')  #Output layer with softmax for multi-class classification
])
# 3. Set the custom learning rate for Adam optimizer
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
# Compile the model with the custom optimizer
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])
# 4. Define a callback to print accuracy every 5 epochs
class PrintAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy = self.model.evaluate(test_data)
            print(f"Epoch {epoch+1}: Test Accuracy: {test_accuracy*100:.2f}%")
# 5. Train the model with the callback
print_accuracy_callback = PrintAccuracyCallback()
history = model.fit(train_data,
                    epochs=EPOCHS,
                    validation_data=test_data,
                    steps_per_epoch=train_data.samples // BATCH_SIZE,
                    validation_steps=test_data.samples // BATCH_SIZE,
                    callbacks=[print_accuracy_callback])
# 6. Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
model.save(MODEL_PATH)
# 7. Visualize the training process
# Plotting the accuracy and loss curves
plt.figure(figsize=(12, 4))
# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()