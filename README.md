# Rethinking Binarized Neural Network Optimization

[![License: Apache 2.0](https://img.shields.io/github/license/CuauSuarez/rethinking-bnn-optimization.svg)](https://github.com/CuauSuarez/BopAndBeyond/blob/conference/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Implementation for paper "[Bop and Beyond: A Second Order Optimizer for Binarized Neural Networks]()".

Great part of this code was adapted from [Bop](https://github.com/plumerai/rethinking-bnn-optimization)

## Requirements

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `2.3.0`
- [Larq](https://github.com/larq/larq) version `0.10.0`
- [Zookeeper](https://github.com/plumerai/zookeeper) version `0.5.5`

You can also check out one of our prebuilt [docker images](https://hub.docker.com/r/plumerai/deep-learning/tags).

## Installation

This is a complete Python module. To install it in your local Python environment, `cd` into the folder containing `setup.py` and run:

```
pip install -e .
```

## Train

To train a model locally, you can use the cli:

```
bnno train binarynet --dataset cifar10
```

## Reproduce Paper Experiments

### Biased or Unbiased. Batch or Layer Normalization (section 5.1.1)

To reproduce the runs comparing the different combinations of:

1. Batch & Biased

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder \
    --hparams threshold=1e-6,gamma=1e-7,sigma=1e-3,epochs=150 \
    --tensorboard=True
```

2. Batch & Unbiased

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_unbiased \
    --hparams threshold=1e-6,gamma=1e-7,sigma=1e-3,epochs=150 \
    --tensorboard=True
```

3. Layer & Biased

```
bnno train binarynet \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder \
    --hparams threshold=1e-6,gamma=1e-7,sigma=1e-3,epochs=150 \
    --tensorboard=True
```

4. Layer & Unbiased

```
bnno train binarynet \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_unbiased \
    --hparams threshold=1e-6,gamma=1e-7,sigma=1e-3,epochs=150 \
    --tensorboard=True
```


### Hyper-parameters exploration (section 5.1.2)

#### Ablation Studies (section 5.1.3)

To reproduce the runs exploring various hyperparameters, run:

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_unbiased \
    --hparams threshold=1e-6,gamma=1e-7,sigma=1e-3,epochs=100 \
    --tensorboard=True
```

where you use the appropriate values for threshold, gamma, and sigma.

#### The effect of schedulers (section 5.1.4)

To reproduce the exploration of the schedulers applied to the hyperparameters, for the unbiased version, run:

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_unbiased_testExp \
    --hparams epochs_decay=100,threshold=1e-6,threshold_decay=0.1,gamma=1e-7,gamma_decay=0.1,sigma=1e-3,sigma_decay=0.1 \
    --tensorboard=True
```

For the biased version:

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_testExp \
    --hparams epochs_decay=100,threshold=1e-6,threshold_decay=0.1,gamma=1e-7,gamma_decay=0.1,sigma=1e-3,sigma_decay=0.1 \
    --tensorboard=True
```

where you use the appropriate initial values for the hyperparameters, the decay values for each of them, and at how many epochs will the values be decayed.


### CIFAR-10 (section 5.2)

To achieve the accuracy in the paper of 91.4% for the unbiased version, run:

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_unbiased_CIFAR \
    --tensorboard=True
```

To achieve the accuracy in the paper of 91.9% for the biased version, run:

```
bnno train binarynet_batch \
    --dataset cifar10 \
    --preprocess-fn resize_and_flip \
    --hparams-set bop2ndOrder_CIFAR \
    --tensorboard=True
```

### ImageNet (section 5.3)

To reproduce the reported results on ImageNet, run:

```
bnno train xnornet_batch --dataset imagenet2012 --hparams-set bop2ndOrder_ImageNet
bnno train birealnet_batch --dataset imagenet2012 --hparams-set bop2ndOrder_ImageNet
```

This should give the results listed below. Click on the tensorboard icons to see training and validation accuracy curves of the reported runs.

<table>
  <tr>
    <th>Network</th>
    <th colspan="2">Bop2ndOrder - top-1 accuracy</th>
  </tr>
  <tr>
    <td>XNOR-Net</td>
    <td>46.9%</td>
    <td>
      <a
        href="https://tensorboard.dev/experiment/IecdQWj3SWOLsmj1NKB8AQ"
        ><img
          src="https://user-images.githubusercontent.com/29484762/68027986-af2bc800-fcab-11e9-94a3-78d8aae7688b.png"
          alt="tensorboard"
      /></a>
    </td>
  </tr>
  <tr>
    <td>Bi-Real Net</td>
    <td>57.2%</td>
    <td>
      <a
        href="https://tensorboard.dev/experiment/UdLm9G41T0GEhNN7ohDAqQ"
        rel="nofollow"
        ><img
          src="https://user-images.githubusercontent.com/29484762/68027986-af2bc800-fcab-11e9-94a3-78d8aae7688b.png"
          alt="tensorboard"
      /></a>
    </td>
  </tr>
</table>
