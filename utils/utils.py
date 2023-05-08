import dataloaders
import models.mapping
import argparse


def update_config(
    config, name=None, model=None
):
    if model:
        set_model(config, model)
    if name:
        set_model_name(config, name)
    

def update_config_from_cli(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Model name")
    parser.add_argument(
        "-m", "--model", default="glavitu", 
        choices=["glavitu", "setr", "resunet", "transunet"], 
        help="Model architecture"
    )
    args = parser.parse_args()
    update_config(
        config, 
        name=args.name,
        model=args.model,
    )


def build_dataloader(sampler_builder, sampler_args, plugins, batch_size, labels, len_factor=1):
    sampler = sampler_builder(**sampler_args)
    return dataloaders.DataLoader(sampler, plugins, batch_size, labels, len_factor)


def build_model(model_builder, model_args, data_config=None, weights_path=None, mode="training"):
    if mode not in {"training", "testing"}:
        raise ValueError()
    # initialize input_shapes and n_outputs for the model
    if data_config:
        model_args["input_shapes"] = data_config.input_shapes
        model_args["n_outputs"] = data_config.n_outputs
    # build the model
    model = model_builder(**model_args)
    # load weights if requested
    if weights_path:
        if isinstance(model, tuple):
            model[0].load_weights(weights_path)
        else:
            model.load_weights(weights_path)
    # choose a proper subnet for the mode
    if isinstance(model, tuple):
        if mode == "training":
            model = model[0]
        else:
            model = model[1]
    return model


def set_model(config, model):
    if model == "glavitu":
        set_model_name(config, "GlaViTU")
        config.model.model_builder = models.mapping.GlaViTU
        config.model.n_ds_branches = 2
    elif model == "resunet":
        set_model_name(config, "ResUNet")
        config.model.model_builder = models.mapping.FusionResUNet
        config.model.n_ds_branches = 1
    elif model == "setr":
        set_model_name(config, "SETR_B16")
        config.model.model_builder = models.mapping.FusionSETR_B16
        config.model.n_ds_branches = 1
    elif model == "transunet":
        set_model_name(config, "TransUNet")
        config.model.model_builder = models.mapping.TransUNet
        config.model.n_ds_branches = 1
    else:
        raise NotImplementedError()


def set_model_name(config, model_name):
    config.model.model_name = model_name
    config.model.model_args.update({
        "name": model_name 
    })
