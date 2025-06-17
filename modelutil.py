# import os 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# '''def load_model() -> Sequential: 
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(TimeDistributed(Flatten()))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

#     model.load_weights(os.path.join('..','models','checkpoint'))

#     return model'''
    
# import os
# import sys
# print("Python executable being used:", sys.executable)
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 's1')

# print("Resolved data path:", DATA_DIR)


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional,
#                                      MaxPool3D, Activation, TimeDistributed, Flatten)

# def load_model() -> Sequential:
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(TimeDistributed(Flatten()))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

#     # Load TensorFlow-style checkpoint
#     checkpoint_path = os.path.join('..', 'models-checkpoint 96', 'checkpoint 96')  # Prefix name
#     checkpoint = tf.train.Checkpoint(model=model)
#     checkpoint.restore(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))).expect_partial()

#     # Optional: Save to .h5 for future reuse with Keras 3
#     model.save_weights(os.path.join('..', 'models-checkpoint 96', 'converted_model.weights.h5'))

#     return model


'''import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional,
                                     MaxPool3D, Activation, TimeDistributed, Flatten)

def load_model() -> Sequential:
    # Define the model architecture
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Loading weights: choose one of the options depending on the model's saved format
    # Option 1: Load TensorFlow-style checkpoint
    checkpoint_path = os.path.join('..', 'models-checkpoint 96', 'checkpoint 96')  # Adjust this path accordingly
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))).expect_partial()
    print(f"Model loaded from checkpoint: {checkpoint_path}")

    # Option 2: If you have a .h5 model weight file, you can load it with:
    # model.load_weights(os.path.join('..', 'models-checkpoint 96', 'converted_model.weights.h5'))

    # Optionally save weights as .h5 for future reuse with Keras 3
    model.save_weights(os.path.join('..', 'models-checkpoint 96', 'converted_model.weights.h5'))

    return model

# Debugging: Check if the model is loaded correctly
if __name__ == "__main__":
    model = load_model()
    model.summary()  # Check the model architecture'''
    
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional,
                                     MaxPool3D, Activation, TimeDistributed, Flatten)

def load_model(checkpoint_dir='../models-checkpoint 96') -> Sequential:
    # Define the model architecture
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Load the latest checkpoint from the directory
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"✅ Model weights loaded from: {latest_ckpt}")
    else:
        print("❌ No checkpoint found in:", checkpoint_dir)

    return model

# Debugging: Check if the model is loaded correctly
if __name__ == "__main__":
    model = load_model('../models-checkpoint 96')  # or use '../models-checkpoint 50'
    model.summary()
