from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def binarynet(hparams, input_shape, num_classes):
    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer=hparams.kernel_quantizer,
        kernel_constraint=hparams.kernel_constraint,
        use_bias=False,
    )
    return tf.keras.models.Sequential(
        [
            # don't quantize inputs in first layer
            lq.layers.QuantConv2D(
                hparams.filters,
                hparams.kernel_size,
                kernel_quantizer=hparams.kernel_quantizer,
                kernel_constraint=hparams.kernel_constraint,
                use_bias=False,
                input_shape=input_shape,
            ),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Flatten(),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            lq.layers.QuantDense(num_classes, **kwhparams),
            #tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(binarynet)
class default(HParams):
    epochs = 100
    filters = 128
    dense_units = 1024
    kernel_size = 3
    batch_size = 256
    optimizer = tf.keras.optimizers.Adam(5e-3)
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@registry.register_hparams(binarynet)
class bop(default):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    gamma = 1e-3
    lr = 0.01

    '''
    @property
    def optimizer(self):
        return optimizers.Bop(
            fp_optimizer=tf.keras.optimizers.Adam(self.lr),
            threshold=self.threshold,
            gamma=self.gamma,
        )
    '''
      
    @property
    def optimizer(self):
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=self.threshold,
                    gamma=self.gamma,
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(self.lr),  # for FP weights
        )
        
@registry.register_hparams(binarynet)
class bop2ndOrder(default):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    gamma = 1e-6
    sigma = 1e-2
    lr = 0.01

    '''
    @property
    def optimizer(self):
        return optimizers.Bop2ndOrder(
            tf.keras.optimizers.Adam(self.lr), 
            threshold=self.threshold, 
            gamma=self.gamma, 
            gamma2=self.gamma2
        )
        
    '''
    
    @property
    def optimizer(self):
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder.is_binary_variable, 
                optimizers.Bop2ndOrder(
                    threshold=self.threshold,
                    gamma=self.gamma,
                    sigma=self.sigma,
                    name="Bop2ndOrder"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(self.lr),  # for FP weights
        )



@registry.register_hparams(binarynet)
class bop_sec52(default):
    epochs = 500
    batch_size = 50
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-8
    gamma = 1e-4
    gamma_decay = 0.1
    decay_step = int((50000 / 50) * 100)
    lr = 0.01

    '''
    @property
    def optimizer(self):
        return optimizers.Bop(
            fp_optimizer=tf.keras.optimizers.Adam(0.01),
            threshold=self.threshold,
            gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                self.gamma, self.decay_step, self.gamma_decay, staircase=True
            ),
        )
    '''
        
    @property
    def optimizer(self):
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=self.threshold,
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, self.decay_step, self.gamma_decay, staircase=True
                    ),
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(self.lr),  # for FP weights
        )
