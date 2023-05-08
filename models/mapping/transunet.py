import tensorflow as tf
import layers.general
import models.mapping
import models.misc


def TransUNet(
    input_shapes, n_outputs, last_activation="softmax", dropout=0,
    name="TransUNet", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock(
        64, 
        spatial_dropout=dropout, 
    )(inputs)

    encoded1 = models.mapping.ResUNetEncoder(
        fused.shape[1:], 
        n_steps=3, 
        start_n_filters=64,
        dropout=dropout,
    )(fused)

    encoded1_downsampled = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(encoded1[-1])
    encoded2 = models.misc.ViTEncoder(
        encoded1_downsampled.shape[1:], 
        use_class_token=False, 
        patch_size=1, 
        embedding_size=64, 
        mlp_size=256,
        n_blocks=12, 
        n_heads=4, 
        dropout=dropout,
    )(encoded1_downsampled)
    encoded2 = tf.keras.layers.Dense(encoded1_downsampled.shape[-1])(encoded2)
    encoded2 = tf.keras.layers.UpSampling2D(size=(2, 2))(encoded2)

    decoded = models.mapping.ResUNetDecoder(
        encoded2.shape[1:],
        n_outputs=64, 
        n_steps=3, 
        last_activation="linear",
        dropout=dropout,
    )([encoded2] + encoded1[:-1][::-1])

    decoded = tf.keras.layers.Add()([decoded, fused])
    outputs = tf.keras.layers.Dense(
        n_outputs, activation=last_activation, name="output"
    )(decoded)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
