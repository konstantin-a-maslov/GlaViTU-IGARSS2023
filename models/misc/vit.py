import tensorflow as tf
import layers.transformers


def ViTEncoder(
    input_shape, use_class_token=True, patch_size=16, embedding_size=768, mlp_size=3072,
    n_blocks=12, n_heads=12, dropout=0.1, name="ViTEncoder", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)

    patches = layers.transformers.PatchExtraction(patch_size)(inputs)
    patches = layers.transformers.PatchEmbedding(embedding_size)(patches)
    if use_class_token:
        patches = layers.transformers.AddClassToken()(patches)
    patches = layers.transformers.AddPositionalEmbedding()(patches)
    for _ in range(n_blocks):
        patches = layers.transformers.TransformerBlock(
            embedding_size, mlp_size, n_heads, dropout
        )(patches)
    if use_class_token:
        outputs = layers.transformers.ExtractClassToken()(patches)
    else:
        input_height, input_width, _ = input_shape
        outputs = tf.keras.layers.Reshape(
            (input_height // patch_size, input_width // patch_size, -1)
        )(patches)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_size, n_classes, dropout):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(mlp_size, activation=tf.keras.activations.tanh)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax)
        
    def call(self, inputs):
        hidden = self.dense1(inputs)
        hidden = self.dropout(hidden)
        outputs = self.dense2(hidden)
        return outputs


def ViT(
    input_shape, n_classes, patch_size=16, embedding_size=768, mlp_size=3072,
    n_blocks=12, n_heads=12, dropout=0.1, name="ViT", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)
    encoded = ViTEncoder(
        input_shape,
        use_class_token=True,
        patch_size=patch_size, 
        embedding_size=embedding_size, 
        mlp_size=mlp_size,
        n_blocks=n_blocks, 
        n_heads=n_heads, 
        dropout=dropout,
    )(inputs)
    outputs = MLP(mlp_size, n_classes, dropout)(encoded)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
