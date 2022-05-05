import tensorflow as tf

class AdvLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)
    def call(self, inputs):
        ori_feature, gan_feature = inputs
        loss = tf.math.reduce_mean(tf.math.square(ori_feature - tf.math.reduce_mean(gan_feature, axis=0)))
        return loss
    
class CntLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)
    def call(self, inputs):
        x, x_ = inputs
        loss = tf.math.reduce_mean(tf.math.abs(x - x_))
        return loss
    
class EncLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)
    def call(self, inputs):
        z, z_ = inputs
        loss = tf.math.reduce_mean(tf.math.square(z - z_))
        return loss

class GANomaly(AdvLoss, CntLoss, EncLoss):
    def __init__(self):
        self.Ge = self.encoder()
        self.Gd = self.decoder()
        self.E = self.encoder()
        self.D = self.feature_extractor()

    def encoder(self):
        inputs = tf.keras.Input((28, 28, 1))                                 # [28, 28,  1]
        x = tf.keras.layers.Conv2D( 2, 7, kernel_regularizer = 'l2')(inputs) # [22, 22,  2]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D( 4, 7, kernel_regularizer = 'l2')(x)      # [16, 16,  4]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D( 8, 7, kernel_regularizer = 'l2')(x)      # [10, 10,  8]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(16, 7, kernel_regularizer = 'l2')(x)      # [ 4,  4, 16]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
        return tf.keras.Model(inputs, outputs)

    def decoder(self):
        inputs = tf.keras.Input((16,))
        x = tf.expand_dims(inputs, axis=1)
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, multiples=[1, 4, 4, 1])                                   # [ 4,  4, 16]
        x = tf.keras.layers.Conv2DTranspose( 8, 7, kernel_regularizer = 'l2')(x) # [10, 10,  8]
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose( 4, 7, kernel_regularizer = 'l2')(x) # [16, 16,  4]
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose( 2, 7, kernel_regularizer = 'l2')(x) # [22, 22,  2]
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose( 1, 7, kernel_regularizer = 'l2')(x) # [28, 28,  1]
        outputs = tf.keras.layers.Activation('tanh')(x)
        return tf.keras.Model(inputs, outputs)

    def feature_extractor(self):
        inputs = tf.keras.Input((28, 28, 1))       # [28, 28,  1]
        x = tf.keras.layers.Conv2D( 2, 7, kernel_regularizer = 'l2')(inputs)   # [22, 22,  2]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D( 4, 7, kernel_regularizer = 'l2')(x)        # [16, 16,  4]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D( 8, 7, kernel_regularizer = 'l2')(x)        # [10, 10,  8]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(16, 7, kernel_regularizer = 'l2')(x)        # [ 4,  4, 16]
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs, outputs)

    def loss(self, y_true, y_pred):
        return y_pred

    def get_model(self):
        inputs = tf.keras.Input((28, 28, 1))
        x = inputs
        # Autoencoder
        z = self.Ge(x)
        x_ = self.Gd(z)
        # Encoder network
        z_ = self.E(x_)
        # Discriminator network
        ori_feature = self.D(x)
        gan_feature = self.D(x_)
        # Losses
        adv_loss = AdvLoss(name='adv_loss')([ori_feature, gan_feature])
        cnt_loss = CntLoss(name='cnt_loss')([x, x_])
        enc_loss = EncLoss(name='enc_loss')([z, z_])
        losses = {
            'adv_loss': self.loss,
            'cnt_loss': self.loss,
            'enc_loss': self.loss,
        }
        # loss_weights
        loss_weights = {'adv_loss': 1.0, 'cnt_loss': 50.0, 'enc_loss': 1.0}
        
        outputs = [adv_loss, cnt_loss, enc_loss]
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
        return model

    def get_G_model(self):
        # Autoencoder
        inputs = tf.keras.Input((28, 28, 1))
        z = self.Ge(inputs)
        outputs = self.Gd(z)
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_D_model(self):
        inputs = tf.keras.Input((28, 28, 1))
        x = self.D(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

if __name__ == "__main__":
    ganomaly = GANomaly()
    model = ganomaly.get_model()
    print(model.summary())
