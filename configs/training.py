import tensorflow as tf
import losses
import metrics
import utils.deeplearning
import os


learning_rate = 5e-4
max_epochs = 1000
optimizer = tf.keras.optimizers.Adam()

gamma = 2.0


def get_loss(n_ds_branches=1):
    loss = [
        losses.FocalLoss(gamma=gamma) for _ in range(n_ds_branches)
    ]
    if n_ds_branches == 1:
        loss = loss[0]
    return loss


def get_metrics(n_outputs, n_ds_branches=1):
    metrics_list = [
        [
            tf.keras.metrics.CategoricalAccuracy(),
            *[
                tf.keras.metrics.Precision(class_id=class_idx, name=f"precision_{class_idx}") 
                for class_idx in range(n_outputs)
            ],
            *[
                tf.keras.metrics.Recall(class_id=class_idx, name=f"recall_{class_idx}") 
                for class_idx in range(n_outputs)
            ],
            *[
                metrics.IoU(
                    class_id=class_idx, 
                    name=f"iou_{class_idx}" if n_ds_branches > 1 else f"output_iou_{class_idx}"
                ) 
                for class_idx in range(n_outputs)
            ],
        ] 
        for _ in range(n_ds_branches)
    ]
    if n_ds_branches == 1:
        metrics_list = metrics_list[0]
    return metrics_list


def get_callbacks(model_name, steps_per_epoch, n_outputs=2):
    callbacks = [
        utils.deeplearning.LRWarmup(
            warmup_steps=steps_per_epoch,
            target=learning_rate,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_output_iou_1",
            mode="max",
            factor=0.1,
            patience=10,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_output_iou_1",
            mode="max",
            patience=31,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("weights", f"{model_name}_weights.h5"),
            monitor=f"val_output_iou_1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join("logs", f"{model_name}_log.csv"),
        )
    ]
    return callbacks
