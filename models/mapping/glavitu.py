import tensorflow as tf
import layers.general
import models.mapping
import models.misc


def GlaViTU(
    input_shapes, n_outputs, last_activation="softmax", dropout=0,
    name="GlaViTU", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock(
        64, 
        spatial_dropout=dropout, 
    )(inputs)

    encoded1 = models.mapping.SETR(
        fused.shape[1:],
        n_outputs=64,
        last_activation="linear",
        patch_size=16,
        embedding_size=64,
        mlp_size=256,
        n_blocks=12,
        n_heads=4,
        dropout=dropout,
        n_filters=64,
    )(fused)
    encoded1 = tf.keras.layers.Add()([encoded1, fused])
    outputs1 = tf.keras.layers.Dense(
        n_outputs, activation=last_activation, name="deepsupervision"
    )(encoded1)

    encoded2 = models.mapping.ResUNet(
        encoded1.shape[1:],
        n_outputs=64,
        n_steps=3,
        start_n_filters=64,
        last_activation="linear",
        dropout=dropout,
    )(encoded1)
    encoded = tf.keras.layers.Add()([encoded1, encoded2])
    outputs2 = tf.keras.layers.Dense(
        n_outputs, activation=last_activation, name="output"
    )(encoded)

    train_model = tf.keras.models.Model(inputs=inputs, outputs=[outputs1, outputs2], name=name, **kwargs)
    test_model = tf.keras.models.Model(inputs=inputs, outputs=outputs2, name=name, **kwargs)
    return train_model, test_model
