from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def xnornet(hparams, input_shape, num_classes):

    kwargs = dict(
        kernel_quantizer=hparams.kernel_quantizer,
        input_quantizer="ste_sign",
        kernel_constraint=None,
        use_bias=False,
    )
    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        96,
        (11, 11),
        strides=(4, 4),
        padding="same",
        use_bias=False,
        input_shape=input_shape,
        kernel_regularizer=hparams.kernel_regularizer,
    )(img_input)

    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-5)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **kwargs)(x)
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # Equivalent to a dense layer
    x = lq.layers.QuantConv2D(4096, (1, 1), strides=(1, 1), padding="valid", **kwargs)(
        x
    )
    #x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-3)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        num_classes, use_bias=False, kernel_regularizer=hparams.kernel_regularizer
    )(x)
    x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.models.Model(img_input, x)


@registry.register_hparams(xnornet)
class bop(HParams):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 1281167
    batch_size = 1024
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)

    threshold = 1e-8

    gamma_start = 1e-4
    gamma_end = 1e-6

    lr_start = 2.5e-3
    lr_end = 5e-6

    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)

    @property
    def optimizer(self):
        decay_step = self.epochs_decay * self.train_samples // self.batch_size
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_step, end_learning_rate=self.lr_end, power=1.0
        )
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_step, end_learning_rate=self.gamma_end, power=1.0
        )
        
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=self.threshold,
                    gamma=gamma,
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        ) 


@registry.register_hparams(xnornet)
class bop2ndOrder(HParams):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    gamma = 1e-6
    sigma = 1e-2
    lr = 0.01

    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)

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
        
        
@registry.register_hparams(xnornet)
class bop2ndOrder_unbiased(HParams):
    batch_size = 100
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    threshold = 1e-6
    gamma = 1e-6
    sigma = 1e-2
    lr = 0.01


    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)

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
        
###############################################################################################

@registry.register_hparams(xnornet)
class bop_testExp(HParams):
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
    
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)
    
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
        
        

@registry.register_hparams(xnornet)
class bop_testPoly(HParams):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 1281167
    batch_size = 100
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-6
    threshold_end = 1e-8
    
    
    gamma_start = 1e-7
    gamma_end = 1e-8

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)
    
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

@registry.register_hparams(xnornet)
class bop2ndOrder_testExp(HParams):
    epochs = 300
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold = 1e-5
    threshold_decay = 0.1
    
    gamma = 1e-7
    gamma_decay = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    
    sigma = 1e-3
    sigma_decay = 0.1

    
    lr = 0.01
    lr_decay = 0.1
        
        
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)    
    
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
        


@registry.register_hparams(xnornet)
class bop2ndOrder_testPoly(HParams):
    epochs = 300
    epochs_decay = 300
    
    train_samples = 1281167
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-7
    
    
    gamma_start = 1e-7
    gamma_end = 1e-8
    
    sigma_start = 1e-3
    sigma_end = 1e-5
    

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)
    
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

@registry.register_hparams(xnornet)
class bop2ndOrder_unbiased_testExp(HParams):
    epochs = 300
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold = 1e-5
    threshold_decay = 0.1
    
    gamma = 1e-7
    gamma_decay = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    
    sigma = 1e-3
    sigma_decay = 0.1

    
    lr = 0.01
    lr_decay = 0.1
        
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)
    
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
                    self.lr, self.decay_step, self.lr_decay, staircase=True
                ),
            ),  # for FP weights
        )
        


@registry.register_hparams(xnornet)
class bop2ndOrder_unbiased_testPoly(HParams):
    epochs = 300
    epochs_decay = 300
    
    train_samples = 1281167
    batch_size = 100
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-5
    threshold_end = 1e-7
    
    
    gamma_start = 1e-7
    gamma_end = 1e-8
    
    sigma_start = 1e-3
    sigma_end = 1e-5
    

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)
    
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