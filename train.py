import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Dataset paths
TRAIN_DIR = r'D:\Brain_tumour_detection\dataset\Train'  # Update with the correct path to the new dataset

# Image size for the model
IMG_SIZE = (224, 224)

# Class names
TUMOR_CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% of the training data as validation data
)

# Train data generator with augmentation
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use training data for training
)

# Validation data generator (using the validation split)
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use validation data for validation
)

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(TUMOR_CLASSES), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    model = build_model()

    # Callbacks to save the best model and stop early
    callbacks = [
        ModelCheckpoint('brain_tumor_model.h5', save_best_only=True),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Save the final model
    model.save('brain_tumor_model.h5')  # Save model as .h5 file
    print(f"Model saved to 'brain_tumor_model.h5'")

    return model, history

# Main function to train and save the model
if __name__ == "__main__":
    model, history = train_model()
