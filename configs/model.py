import models.mapping


model_name = "GlaViTU"
dropout = 0.10

model_builder, model_args = models.mapping.GlaViTU, {
    "last_activation": "softmax", 
    "dropout": dropout, 
    "name": model_name
}

n_ds_branches = 2
