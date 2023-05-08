import abc
import dataloaders.transformations
import numpy as np


class Plugin(abc.ABC):
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def get_sampler(self):
        if not self.dataloader:
            raise ValueError()
        return self.dataloader.sampler

    def __init_subclass__(cls, **kwargs):
        cls.has_before_indexing_behaviour = not (cls.before_indexing == Plugin.before_indexing)
        cls.has_after_indexing_behaviour = not (cls.after_indexing == Plugin.after_indexing)
        cls.has_on_sampling_behaviour = not (cls.on_sampling == Plugin.on_sampling)
        cls.has_on_finalising_behaviour = not (cls.on_finalising == Plugin.on_finalising)

    def before_indexing(self, sampler):
        pass

    def after_indexing(self, sampler):
        pass

    def on_sampling(self, sample):
        return sample

    def on_finalising(self, batch_x, batch_y):
        return batch_x, batch_y


class Augmentation(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def on_sampling(self, sample):
        dataloaders.transformations.apply_transformations(
            sample, self.transformations
        )
        return sample
