from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def birealnet_batch(hparams, input_shape, num_classes):
    def residual_block(x, double_filters=False, filters=None):
        assert not (double_filters and filters)
        
        no_op = lq.quantizers.NoOpQuantizer(precision=1)

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters, 1, kernel_initializer="glorot_normal", use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)
            #shortcut = tf.keras.layers.LayerNormalization()(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            3,
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer="approx_sign",
            kernel_quantizer=no_op,
            kernel_initializer="glorot_normal",
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        #x = tf.keras.layers.LayerNormalization()(x)
        return tf.keras.layers.add([x, shortcut])

    img_input = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        64,
        7,
        strides=2,
        kernel_initializer="glorot_normal",
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    #out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)

    # layer 2
    out = residual_block(out, filters=64)

    # layer 3 - 5
    for _ in range(3):
        out = residual_block(out)

    # layer 6 - 17
    for _ in range(3):
        out = residual_block(out, double_filters=True)
        for _ in range(3):
            out = residual_block(out)

    # layer 18
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)



@registry.register_hparams(birealnet_batch)
class bopBase(HParams):
    epochs = 400
    batch_size = 256

    threshold = 1e-8

    gamma = 1e-5
    

    lr_start = 2.5e-3
    lr_end = 5e-6

    @property
    def optimizer(self):
        decay_step = self.epochs * 1281167 // self.batch_size
        
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_step, end_learning_rate=self.lr_end, power=1.0
        )
        
        return lq.optimizers.CaseOptimizer(
            (optimizers.Bop.is_binary_variable, 
                optimizers.Bop(
                    threshold=self.threshold,
                    gamma=self.gamma,
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(lr),  # for FP weights
        ) 


@registry.register_hparams(birealnet_batch)
class bop(HParams):
    epochs = 300
    epochs_decay = 300
    
    train_samples = 1281167
    batch_size = 1024

    threshold = 1e-8

    gamma_start = 1e-4
    gamma_end = 1e-6
    

    lr_start = 2.5e-3
    lr_end = 5e-6

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
        
    
@registry.register_hparams(birealnet_batch)
class bop2ndorder(HParams):
    epochs = 300
    batch_size = 256

    threshold = 1e-8

    gamma = 1e-5
    sigma = 1e-6

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


@registry.register_hparams(birealnet_batch)
class bop2ndorder_unbiased(HParams):
    epochs = 300
    
    batch_size = 256

    threshold = 1e-8

    gamma = 1e-5
    sigma = 1e-6

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


###############################################################################################

@registry.register_hparams(birealnet_batch)
class bop_testExp(HParams):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50

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
                        self.threshold, self.decay_step, self.threshold_decay, staircase=True
                    ),
                    gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                        self.gamma, self.decay_step, self.gamma_decay, staircase=True
                    ),
                    name="Bop"
                )
            ),
            default_optimizer=tf.keras.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr, self.decay_step, self.lr_decay, staircase=True
                ),
            ),  # for FP weights
        )
        
        

@registry.register_hparams(birealnet_batch)
class bop_testPoly(HParams):
    epochs = 100
    epochs_decay = 100
    
    train_samples = 1281167
    batch_size = 50
    
    
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

@registry.register_hparams(birealnet_batch)
class bop2ndOrder_testExp(HParams):
    epochs = 300
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 100
    
    threshold = 1e-5
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
        


@registry.register_hparams(birealnet_batch)
class bop2ndOrder_testPoly(HParams):
    epochs = 150
    epochs_decay = 150
    
    train_samples = 1281167
    batch_size = 1024
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-8
    threshold_end = 1e-5
    
    
    gamma_start = 1e-4
    gamma_end = 1e-9
    
    sigma_start = 1e-5
    sigma_end = 1e-2
    

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    regularization_quantity = 5e-7
    
    
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



@registry.register_hparams(birealnet_batch)
class bop2ndOrder_ImageNet(HParams):
    epochs = 150
    epochs_decay = 150
    
    train_samples = 1281167
    batch_size = 1024
    
    kernel_quantizer = lq.quantizers.NoOpQuantizer(precision=1)
    kernel_constraint = None
    
    threshold_start = 1e-8
    threshold_end = 1e-5
    
    
    gamma_start = 1e-4
    gamma_end = 1e-9
    
    sigma_start = 1e-5
    sigma_end = 1e-2
    

    lr_start = 2.5e-3
    lr_end = 5e-6
    
    regularization_quantity = 5e-7
    
    
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

@registry.register_hparams(birealnet_batch)
class bop2ndOrder_unbiased_testExp(HParams):
    epochs = 300
    epochs_decay = 100
    
    train_samples = 50000
    batch_size = 50
    
    threshold = 1e-5
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
        


@registry.register_hparams(birealnet_batch)
class bop2ndOrder_unbiased_testPoly(HParams):
    epochs = 300
    epochs_decay = 300
    
    train_samples = 1281167
    batch_size = 50
    
    
    threshold_start = 1e-5
    threshold_end = 1e-7
    
    
    gamma_start = 1e-7
    gamma_end = 1e-8
    
    sigma_start = 1e-3
    sigma_end = 1e-5
    

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