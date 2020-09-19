from copy import deepcopy
from typing import Callable, Optional, Tuple

import tensorflow as tf

import larq as lq
from larq import utils

__all__ = ["Bop", "Bop2ndOrder"]



@utils.register_keras_custom_object
class Bop(tf.keras.optimizers.Optimizer):
    """Binary optimizer (Bop).
    Bop is a latent-free optimizer for Binarized Neural Networks (BNNs) and
    Binary Weight Networks (BWN).
    Bop maintains an exponential moving average of the gradients controlled by
    `gamma`. If this average exceeds the `threshold`, a weight is flipped.
    The hyperparameter `gamma` is somewhat analogues to the learning rate in
    SGD methods: a high `gamma` results in rapid convergence but also makes
    training more noisy.
    Note that the default `threshold` is not optimal for all situations.
    Setting the threshold too high results in little learning, while setting it
    too low results in overly noisy behaviour.
    !!! warning
        The `is_binary_variable` check of this optimizer will only target variables that
        have been explicitly marked as being binary using `NoOp(precision=1)`.
    !!! example
        ```python
        no_op_quantizer = lq.quantizers.NoOp(precision=1)
        layer = lq.layers.QuantDense(16, kernel_quantizer=no_op_quantizer)
        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, lq.optimizers.Bop()),
            default_optimizer=tf.keras.optimizers.Adam(0.01),  # for FP weights
        )
        ```
    # Arguments
        threshold: magnitude of average gradient signal required to flip a weight.
        gamma: the adaptivity rate.
        name: name of the optimizer.
    # References
        - [Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization](https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self, threshold: float = 1e-8, gamma: float = 1e-4, name: str = "Bop", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self._set_hyper("threshold", threshold)
        self._set_hyper("gamma", gamma)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _get_decayed_hyper(self, name: str, var_dtype):
        hyper = self._get_hyper(name, var_dtype)
        if isinstance(hyper, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            hyper = tf.cast(hyper(local_step), var_dtype)
        return hyper

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        gamma = self._get_decayed_hyper("gamma", var_dtype)
        threshold = self._get_decayed_hyper("threshold", var_dtype)
        m = self.get_slot(var, "m")

        m_t = m.assign_add(gamma * (grad - m))
        var_t = lq.math.sign(-tf.sign(var * m_t - threshold) * var)
        return var.assign(var_t).op

    def get_config(self):
        config = {
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
        }
        return {**super().get_config(), **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        for hyper in ("gamma", "threshold"):
            if hyper in config and isinstance(config[hyper], dict):
                config[hyper] = tf.keras.optimizers.schedules.deserialize(
                    config[hyper], custom_objects=custom_objects
                )
        return cls(**config)

    @staticmethod
    def is_binary_variable(var: tf.Variable) -> bool:
        """Returns `True` for variables with `var.precision == 1`.
        This is an example of a predictate that can be used by the `CaseOptimizer`.
        # Arguments
            var: a `tf.Variable`.
        """
        return getattr(var, "precision", 32) == 1
        



@utils.register_keras_custom_object
class Bop2ndOrder(tf.keras.optimizers.Optimizer):
    """Second Order Binary optimizer (Bop2ndOrder).
  
    # Arguments
        threshold: magnitude of average gradient signal required to flip a weight.
        gamma: the adaptivity rate.
        sigma: variance scaler rate.
        name: name of the optimizer.
    # References
        - [Bop and Beyond: an Iteration on Binarized Neural Networks Optimization](https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self, 
        threshold: float = 1e-6,
        gamma: float = 1e-6,
        sigma: float = 1e-2,
        name: str = "Bop2ndOrder", 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self._set_hyper("gamma", gamma)
        self._set_hyper("sigma", sigma)
        self._set_hyper("threshold", threshold)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def _get_decayed_hyper(self, name: str, var_dtype):
        hyper = self._get_hyper(name, var_dtype)
        if isinstance(hyper, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            hyper = tf.cast(hyper(local_step), var_dtype)
        return hyper

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        gamma = self._get_decayed_hyper("gamma", var_dtype)
        sigma = self._get_decayed_hyper("sigma", var_dtype)
        threshold = self._get_decayed_hyper("threshold", var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = m.assign_add(gamma * (grad - m))
        v_t = v.assign_add(sigma * (tf.math.square(grad) - v))
        var_t = lq.math.sign(-tf.sign(var * (m_t / (tf.math.sqrt(v_t) + 1e-10)) - threshold) * var)
        return var.assign(var_t).op

    def get_config(self):
        config = {
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
            "sigma": self._serialize_hyperparameter("sigma"),
        }
        return {**super().get_config(), **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        for hyper in ("gamma", "sigma", "threshold"):
            if hyper in config and isinstance(config[hyper], dict):
                config[hyper] = tf.keras.optimizers.schedules.deserialize(
                    config[hyper], custom_objects=custom_objects
                )
        return cls(**config)

    @staticmethod
    def is_binary_variable(var: tf.Variable) -> bool:
        """Returns `True` for variables with `var.precision == 1`.
        This is an example of a predictate that can be used by the `CaseOptimizer`.
        # Arguments
            var: a `tf.Variable`.
        """
        return getattr(var, "precision", 32) == 1