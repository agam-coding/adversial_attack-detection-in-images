from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from src.models.base import ModelProvider

class CnnModelProvider(ModelProvider):
    def get_model(self) -> Model:
        model = Sequential([
            InputLayer([28, 28, 1]),
            Conv2D(filters=48, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.2),
            Conv2D(filters=96, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        return model

class TrainedModelProvider(ModelProvider):
    def get_model(self) -> Model:
        # Use tf.keras API directly to avoid compatibility issues
        print("Creating a new model with TensorFlow 2.19.0 API...")
        
        # Define the model using the functional API
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = Conv2D(filters=48, kernel_size=2, padding='same', activation='relu')(inputs)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=96, kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(10, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model with necessary parameters
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Using standard Adam optimizer for Keras 3
            metrics=['accuracy']
        )
        
        print("Model created and compiled successfully")
        return model