import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow import keras

tfds.disable_progress_bar()
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")

builder = tfds.builder("rock_paper_scissors")
info = builder.info

# fig = tfds.show_examples(info, ds_train)

train_images = np.array([example["image"].numpy()[:, :, 0] for example in ds_train])
train_label = np.array([example["label"].numpy() for example in ds_train])

test_images = np.array([example["image"].numpy()[:, :, 0] for example in ds_test])
test_label = np.array([example["label"].numpy() for example in ds_test])


train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images /= 255
test_images /= 255

model = keras.models.Sequential(
    [
        keras.layers.AveragePooling2D(pool_size=(3, 3)),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.fit(train_images, train_label, epochs=5, batch_size=32)

model.save("./my_model")
