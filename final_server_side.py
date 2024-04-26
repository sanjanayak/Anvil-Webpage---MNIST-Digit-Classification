# %%
from keras.models import load_model
import keras
import anvil.server
import csv


import tensorflow as tf
import anvil.media
import numpy as np
from PIL import Image
import pandas as pd


import io
from io import StringIO
from io import BytesIO

# %%
import anvil.server

anvil.server.connect("server_KG3ZRBSWX3CULD6H3WHUFV2Q-ANWN6SUFQEW7IK2D")
# %%
# Load CNN model
cnn_model = tf.keras.models.load_model('final_cnn.h5')


class ClassToken(tf.keras.layers.Layer):
    def _init_(self):
        super()._init_()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

# %%


@anvil.server.callable
def display_image(file):
    # Read CSV data into a DataFrame
    csv_content = file.get_bytes().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content), header=None)

    df_np = np.array(df)

    if np.max(df_np) <= 1:
        df_np = df_np * 255.0

    img = Image.fromarray(df_np.astype('uint8'))

    # Convert image to bytes
    img_bytes = BytesIO()
    name = 'img'
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    return anvil.BlobMedia("image/png", img_bytes, name=name), df_np

# %%


@anvil.server.callable
def cnn_predict(image_data):
    # Convert image_data to the correct format expected by final_model
    # Predict and return result
    csv_content = image_data.get_bytes().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content), header=None)
    np_array = np.array(df)
    np_array = np_array.reshape(1, 28, 28, 1)
    prediction = cnn_model.predict(np_array)
    predicted_class = np.argmax(prediction, axis=1)
    return str(predicted_class[0])  # Assuming prediction is numpy array

# %%


@anvil.server.callable
def read_about(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        # Read the entire contents of the file
        text = file.read()
        return text

# %%


def build_ViT(n, m, block_size, hidden_dim, num_layers, num_heads, key_dim, mlp_dim, dropout_rate, num_classes):
    # n is number of rows of blocks
    # m is number of cols of blocks
    # block_size is number of pixels (with rgb) in each block

    inp = tf.keras.layers.Input(shape=(n*m, block_size))
    inp2 = tf.keras.layers.Input(shape=(n*m))
    # transform to vectors with different dimension
    mid = tf.keras.layers.Dense(hidden_dim)(inp)
    # the positional embeddings
#     positions = tf.range(start=0, limit=n*m, delta=1)
    # learned positional embedding for each of the n*m possible possitions
    emb = tf.keras.layers.Embedding(input_dim=n*m, output_dim=hidden_dim)(inp2)
    mid = mid + emb  # for some reason, tf.keras.layers.Add causes an error, but + doesn't?
    # create and append class token to beginning of all input vectors
    token = ClassToken()(mid)  # append class token to beginning of sequence
    mid = tf.keras.layers.Concatenate(axis=1)([token, mid])

    for l in range(num_layers):  # how many Transformer Head layers are there?
        ln = tf.keras.layers.LayerNormalization()(mid)  # normalize
        mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)(ln, ln, ln)  # self attention!
        add = tf.keras.layers.Add()([mid, mha])  # add and norm
        ln = tf.keras.layers.LayerNormalization()(add)
        # maybe should be relu...who knows...
        den = tf.keras.layers.Dense(mlp_dim, activation='gelu')(ln)
        den = tf.keras.layers.Dropout(dropout_rate)(den)  # regularization
        # back to the right dimensional space
        den = tf.keras.layers.Dense(hidden_dim)(den)
        den = tf.keras.layers.Dropout(dropout_rate)(den)
        mid = tf.keras.layers.Add()([den, add])  # add and norm again
    ln = tf.keras.layers.LayerNormalization()(mid)
    fl = ln[:, 0, :]  # just grab the class token for each image in batch
    clas = tf.keras.layers.Dense(num_classes, activation='softmax')(
        fl)  # probability that the image is in each category
    mod = tf.keras.models.Model([inp, inp2], clas)
    mod.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return mod


n = 4
m = 4
block_size = 49
hidden_dim = 144
num_layers = 8
num_heads = 8
# usually good practice for key_dim to be hidden_dim//num_heads...this is why we do Multi-Head attention
key_dim = hidden_dim//num_heads
mlp_dim = hidden_dim
dropout_rate = 0.06
num_classes = 10


transformer_model = build_ViT(n, m, block_size, hidden_dim, num_layers,
                              num_heads, key_dim, mlp_dim, dropout_rate, num_classes)


transformer_model.load_weights('final_transformer_model.h5')

# %%


@anvil.server.callable
def transformer_predict(image_data):
    csv_content = image_data.get_bytes().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content), header=None)

    np_array = np.array(df)
    if np.max(np_array) > 1:
        np_array = np_array / 255.0

    x_test_ravel = np.zeros((1, 16, 49))

    for img in range(1):
        ind = 0
        for row in range(4):
            for col in range(4):
                x_test_ravel[img, ind, :] = np_array[(
                    row * 7):((row + 1) * 7), (col * 7):((col + 1) * 7)].ravel()
                ind += 1

    pos_feed = np.array([list(range(16))]*1)
    predicted_output = np.argmax(
        transformer_model.predict([x_test_ravel, pos_feed]), axis=1)
    return str(predicted_output[0])

# %%


@anvil.server.callable
def file_validation(file):
    # Read the CSV file and extract data
    csv_data = file.get_bytes().decode('utf-8')
    reader = csv.reader(io.StringIO(csv_data))
    pixel_rows = list(reader)

    # Convert the pixel values to a numpy array with dtype=float
    try:
        pixel_array = np.array(pixel_rows, dtype=np.float32)
    except ValueError:
        return 'non_numeric'

    if pixel_array.shape != (28, 28):
        return 'dimension_error'

    if not np.all((pixel_array >= 0) & (pixel_array <= 255)):
        return 'range_error'

    return 'success'

anvil.server.wait_forever()