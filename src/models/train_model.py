# Image segmentation with a U-Net-like architecture

# %%
import matplotlib.pyplot as plt
import random
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from PIL import ImageOps
import PIL
from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
import os

# %% Prepare paths of input images and target segmentation masks
input_dir = "../../data/processed/RGB"
target_dir = "../../data/processed/GT"
img_size = (256, 256)
num_classes = 2
batch_size = 8

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

# %% What does one input image and corresponding segmentation mask look like?

# Display input image #7
display(Image(filename=input_img_paths[9]))


# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)

# %% Prepare Sequence class to load & vectorize batches of data


class AerailImages(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) +
                     self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size,
                           color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 0, 255. Devide 255 to make them 0, 1
            y[j] = y[j]/255
        return x, y

# %% Test image loader


# Instantiate data Sequences for each split
data_gen = AerailImages(
    batch_size, img_size, input_img_paths, target_img_paths)

x, y = next(iter(data_gen))
print(x.shape)
print(y.shape)

img = keras.preprocessing.image.array_to_img(x[0, :, :, :])
display(img)

img = PIL.ImageOps.autocontrast(
    keras.preprocessing.image.array_to_img(y[0, :, :, :]))
display(img)

# %% Prepare U-Net Xception-style model


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    # for filters in [64, 128, 256]:
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    # for filters in [256, 128, 64, 32]:
    for filters in [128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs, name="UNet_Xception")
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model


model = get_model(img_size, num_classes)
model.summary()

# %% Set aside a validation split

# Split our img paths into a training and a validation set
val_samples = 10
random.Random(2021).shuffle(input_img_paths)
random.Random(2021).shuffle(target_img_paths)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = AerailImages(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = AerailImages(batch_size, img_size,
                       val_input_img_paths, val_target_img_paths)

# %% Train the model
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.

# model = keras.models.load_model('aerial_segmentation.h5')

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "aerial_segmentation.h5", save_best_only=True,  mode='auto')
]

# Train the model, doing validation at the end of each epoch.
epochs = 200
# history = model.fit(train_gen, epochs=epochs,
#                     validation_data=train_gen, callbacks=callbacks)
#
# Try overfit the data
history = model.fit(train_gen, epochs=epochs,
                    validation_data=train_gen, callbacks=callbacks)

# %% plot history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# %% Visualize predictions
# Generate predictions for all images in the validation set
# Generate predictions for all images in the validation set
# batch_size = 8
# val_gen = AerailImages(batch_size, img_size,
#                        val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(
        keras.preprocessing.image.array_to_img(mask))
    display(img)


# Display results for validation image #10
i = 1

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.

# %%


# %%
