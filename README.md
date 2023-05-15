# GlaViTU: A Hybrid CNN-Transformer for Multi-Regional Glacier Mapping from Multi-Source Data

**Authors:** <br/>
Konstantin A. Maslov, <br/>
Claudio Persello, <br/>
Thomas Schellenberger, <br/>
Alfred Stein

This repository contains materials for our paper presented at IGARSS2023.

Glacier mapping is essential for studying and monitoring the impacts of climate change. 
However, several challenges such as debris-covered ice complicate large-scale glacier mapping in a fully-automated manner. 
This work presents a novel hybrid CNN-transformer model (GlaViTU) for multi-regional glacier mapping. 
Our model outperforms three baseline models&mdash;SETR-B/16, ResU-Net and TransU-Net&mdash;achieving a higher mean IoU of 0.875 and demonstrating better generalization ability. 
The proposed model is also parameter-efficient, with approximately 10 and 3 times fewer parameters than SETR-B/16 and ResU-Net, respectively. 
Our results provide a solid foundation for future studies on the application of deep learning methods for global glacier mapping. 
To facilitate reproducibility, we have shared our data set, codebase and pretrained models here.

To cite the paper/repository, please use the following bib entry.

```
@inproceedings{glavitu,
     title = {GlaViTU: A hybrid CNN-transformer for multi-regional glacier mapping from multi-source data},
     author = {K. A. Maslov and C. Persello and T. Schellenberger and A. Stein},
     booktitle = {Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
     year = {2023},
} 
```

### Important links

Before proceeding, please make sure that you have downloaded [the dataset](https://bit.ly/3pnLMhF). 
If you are interested in trying the pretrained models, you can find them [here](https://bit.ly/3pdwUCt).

### Training a model

To train a model, run
 
```
python train.py [-m <ARCHITECTURE>] [-n <MODEL NAME>]
```

Supported architectures are 'setr', 'resunet', 'transunet' and 'glavitu' (default). 
The model name is also optional, if not specified, it will be automatically inferred from the architecture. 

After the training is finished, you will find a .csv log in ./logs and a .h5 file with the model parameters in ./weights.

### Classifying testing data

To classify the test subset, execute

```
python predict.py [-m <ARCHITECTURE>] [-n <MODEL NAME>]
```

### Evaluating the performance

Finally, to evaluate the model, run

```
python evaluate.py [-n <MODEL NAME>]
```

It will output mean IoU and region-wide IoUs for the model.

<br/>

> If you notice any inaccuracies, mistakes or errors, consider submitting a pull request. For inquiries, contact [k.a.maslov@utwente.nl](mailto:k.a.maslov@utwente.nl).
