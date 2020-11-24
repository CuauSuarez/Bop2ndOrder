from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def binarynet_batch(hparams, input_shape, num_classes):
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
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Flatten(),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            lq.layers.QuantDense(num_classes, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            #tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(binarynet_batch)
class default(HParams):
    epochs = 100
    filters = 128
    dense_units = 1024
    kernel_size = 3
    batch_size = 256
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"





###############################################################################################

@registry.register_hparams(binarynet_batch)
class bop(default):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    gamma = 1e-3
    lr = 0.01
      
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
        
@registry.register_hparams(binarynet_batch)
class bop2ndOrder(default):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-5
    gamma = 1e-7
    sigma = 1e-3
    lr = 0.01
    
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

        
@registry.register_hparams(binarynet_batch)
class bop2ndOrder_unbiased(default):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-5
    gamma = 1e-7
    sigma = 1e-3
    lr = 0.01

    
    @property
    def optimizer(self):
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder_unbiased.is_binary_variable, 
                optimizers.Bop2ndOrder_unbiased(
                    threshold=self.threshold,
                    gamma=self.gamma,
                    sigma=self.sigma,
                    name="Bop2ndOrder_unbiased"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(self.lr),  # for FP weights
        )
        



@registry.register_hparams(binarynet_batch)
class bop_sec52(default):
    epochs = 500
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-8
    gamma = 1e-4
    gamma_decay = 0.1
    
    lr = 0.01
        
    @property
    def optimizer(self):
        decay_step = int((self.train_samples / self.batch_size) * self.epochs_decay)
    
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=self.threshold,
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, decay_step, self.gamma_decay, staircase=True
                    ),
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(self.lr),  # for FP weights
        )





###############################################################################################

@registry.register_hparams(binarynet_batch)
class bop_testExp(default):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    threshold_decay = 0.1
    
    gamma = 1e-7
    gamma_decay = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    

    
    lr = 0.01
    lr_decay = 0.1

    
    @property
    def optimizer(self):
        decay_step = int((self.train_samples / self.batch_size) * self.epochs_decay)
    
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.threshold, decay_step, self.threshold_decay, staircase=True
                    ),
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, decay_step, self.gamma_decay, staircase=True
                    ),
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr, decay_step, self.lr_decay, staircase=True
                ),
            ),  # for FP weights
        )
        
        

@registry.register_hparams(binarynet_batch)
class bop_testPoly(default):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 1281167
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-6
    threshold_end = 1e-8
    
    
    gamma_start = 1e-7
    gamma_end = 1e-8

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    
    @property
    def optimizer(self):
        decay_steps = self.epochs_decay * self.train_samples // self.batch_size
         
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_steps, self.lr_end, power=1.0
        )
        
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_steps, self.gamma_end, power=1.0
        )
        
        threshold = tf.keras.optimizers.schedules.PolynomialDecay(
            self.threshold_start, decay_steps, self.threshold_end, power=1.0
        )
        
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=threshold,
                    gamma=gamma,
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        )





###############################################################################################


@registry.register_hparams(binarynet_batch)
class bop2ndOrder_testExp(default):
    epochs = 350
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold = 1e-6
    threshold_decay = 0.1
    
    gamma = 1e-7
    gamma_decay = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    
    sigma = 1e-3
    sigma_decay = 0.1

    
    lr = 0.01
    lr_decay = 0.1
    
        
        
    @property
    def optimizer(self):
        decay_step = int((self.train_samples / self.batch_size) * self.epochs_decay)
    
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder.is_binary_variable, 
                optimizers.Bop2ndOrder(
                    threshold=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.threshold, decay_step, self.threshold_decay, staircase=True
                    ),
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, decay_step, self.gamma_decay, staircase=True
                    ),
                    sigma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.sigma, decay_step, self.sigma_decay, staircase=True
                    ),
                    name="Bop2ndOrder"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr, decay_step, self.lr_decay, staircase=True
                ),
            ),  # for FP weights
        )
        


@registry.register_hparams(binarynet_batch)
class bop2ndOrder_testPoly(default):
    epochs = 500
    epochs_decay = 500
    
    #train_samples = 1281167
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-8
    
    
    gamma_start = 1e-2
    gamma_end = 1e-5
    
    sigma_start = 1e-7
    sigma_end = 1e-2
    

    lr_start = 0.01
    lr_end = 0.001
    
    
    @property
    def optimizer(self):
        decay_steps = self.epochs_decay * self.train_samples // self.batch_size
         
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_steps, self.lr_end, power=1.0
        )
        
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_steps, self.gamma_end, power=1.0
        )
        
        sigma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.sigma_start, decay_steps, self.sigma_end, power=1.0
        )
        
        threshold = tf.keras.optimizers.schedules.PolynomialDecay(
            self.threshold_start, decay_steps, self.threshold_end, power=1.0
        )
                
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder.is_binary_variable, 
                optimizers.Bop2ndOrder(
                    threshold=threshold,
                    gamma=gamma,
                    sigma=sigma,
                    name="Bop2ndOrder"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        )
        
    
  
@registry.register_hparams(binarynet_batch)
class bop2ndOrder_CIFAR(default):
    epochs = 500
    epochs_decay = 500
    
    #train_samples = 1281167
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-8
    
    
    gamma_start = 1e-2
    gamma_end = 1e-5
    
    sigma_start = 1e-7
    sigma_end = 1e-2
    

    lr_start = 0.01
    lr_end = 0.001
    
    
    @property
    def optimizer(self):
        decay_steps = self.epochs_decay * self.train_samples // self.batch_size
         
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_steps, self.lr_end, power=1.0
        )
        
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_steps, self.gamma_end, power=1.0
        )
        
        sigma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.sigma_start, decay_steps, self.sigma_end, power=1.0
        )
        
        threshold = tf.keras.optimizers.schedules.PolynomialDecay(
            self.threshold_start, decay_steps, self.threshold_end, power=1.0
        )
                
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder.is_binary_variable, 
                optimizers.Bop2ndOrder(
                    threshold=threshold,
                    gamma=gamma,
                    sigma=sigma,
                    name="Bop2ndOrder"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        )  
    
    
    
###############################################################################################


@registry.register_hparams(binarynet_batch)
class bop2ndOrder_unbiased_testExp(default):
    epochs = 350
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold = 1e-6
    threshold_decay = 0.1
    
    gamma = 1e-7
    gamma_decay = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    
    sigma = 1e-3
    sigma_decay = 0.1

    
    lr = 0.01
    lr_decay = 0.1
        
        
    @property
    def optimizer(self):
        decay_step = int((self.train_samples / self.batch_size) * self.epochs_decay)
    
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder_unbiased.is_binary_variable, 
                optimizers.Bop2ndOrder_unbiased(
                    threshold=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.threshold, decay_step, self.threshold_decay, staircase=True
                    ),
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, decay_step, self.gamma_decay, staircase=True
                    ),
                    sigma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.sigma, decay_step, self.sigma_decay, staircase=True
                    ),
                    name="Bop2ndOrder_unbiased"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr, decay_step, self.lr_decay, staircase=True
                ),
            ),  # for FP weights
        )
        


@registry.register_hparams(binarynet_batch)
class bop2ndOrder_unbiased_testPoly(default):
    epochs = 500
    epochs_decay = 500
    
    #train_samples = 1281167
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-8
    
    
    gamma_start = 1e-2
    gamma_end = 1e-5
    
    sigma_start = 1e-7
    sigma_end = 1e-2
    

    lr_start = 0.01
    lr_end = 0.001
    
    
    @property
    def optimizer(self):
        decay_steps = self.epochs_decay * self.train_samples // self.batch_size
         
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_steps, self.lr_end, power=1.0
        )
        
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_steps, self.gamma_end, power=1.0
        )
        
        sigma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.sigma_start, decay_steps, self.sigma_end, power=1.0
        )
        
        threshold = tf.keras.optimizers.schedules.PolynomialDecay(
            self.threshold_start, decay_steps, self.threshold_end, power=1.0
        )
                
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder_unbiased.is_binary_variable, 
                optimizers.Bop2ndOrder_unbiased(
                    threshold=threshold,
                    gamma=gamma,
                    sigma=sigma,
                    name="Bop2ndOrder_unbiased"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        )
        

@registry.register_hparams(binarynet_batch)
class bop2ndOrder_unbiased_CIFAR(default):
    epochs = 500
    epochs_decay = 500
    
    #train_samples = 1281167
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-8
    
    
    gamma_start = 1e-2
    gamma_end = 1e-5
    
    sigma_start = 1e-7
    sigma_end = 1e-2
    

    lr_start = 0.01
    lr_end = 0.001
    
    
    @property
    def optimizer(self):
        decay_steps = self.epochs_decay * self.train_samples // self.batch_size
         
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_steps, self.lr_end, power=1.0
        )
        
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_steps, self.gamma_end, power=1.0
        )
        
        sigma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.sigma_start, decay_steps, self.sigma_end, power=1.0
        )
        
        threshold = tf.keras.optimizers.schedules.PolynomialDecay(
            self.threshold_start, decay_steps, self.threshold_end, power=1.0
        )
                
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop2ndOrder_unbiased.is_binary_variable, 
                optimizers.Bop2ndOrder_unbiased(
                    threshold=threshold,
                    gamma=gamma,
                    sigma=sigma,
                    name="Bop2ndOrder_unbiased"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        )