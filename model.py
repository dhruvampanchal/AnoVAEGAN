"""
    AnoVAEGAN model implemented.
"""

#Import functions.
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Conv2DTranspose, LeakyReLU, InputLayer, Flatten, Reshape, Activation, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import metrics, backend as K
import time
import random
import cv2 as cv
import datetime
from tensorflow.keras import mixed_precision
from tensorflow.python.ops import math_ops

mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    class CVAE(tf.keras.Model):
        def __init__(self, input_shape, latent_dim = None):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            
            self.encoder = self.encoder_func(input_shape, custom_bottleneck_size = self.latent_dim)
            self.decoder = self.decoder_func(custom_bottleneck_size = latent_dim)
            
        
        def encoder_func(self, input_shape, custom_bottleneck_size = None):
            inputx = Input(shape = input_shape)
            layer = self.convLayer(inputx, 64) #128
            layer = self.convLayer(layer, 64) #64
            layer = self.convLayer(layer, 128) #32
            layer = self.convLayer(layer, 256) #16
            layer = self.convLayer(layer, 512) #8
            layer = self.convLayer(layer, 512) #4
            layer = Flatten()(layer)
            if (custom_bottleneck_size == None):
                # layer = Conv2D(filters = 16000, kernel_size = layer.shape[1], strides = layer.shape[2], padding = 'same', name = 'intermediate_conv')(layer)
                layer = Dense(16000, dtype = tf.float32)(layer)
            else:
                # layer = Conv2D(filters = custom_bottleneck_size, kernel_size = layer.shape[1], strides = layer.shape[2], padding = 'same', name = 'intermediate_conv')(layer)
                layer = Dense(custom_bottleneck_size + custom_bottleneck_size, dtype = tf.float32)(layer)
            return Model(inputs = inputx, outputs = layer)
        
        
        def decoder_func(self, custom_bottleneck_size = None):
            if (custom_bottleneck_size == None):
                inputx = Input(shape = (8000, ))
            else:
                inputx = Input(shape = (custom_bottleneck_size, ))
            layer = Dense(4*4*512)(inputx)
            layer = LeakyReLU()(layer)
            layer = Reshape(target_shape = (4, 4, 512))(layer)
            layer = self.deConvLayer(layer, 512) #8
            layer = self.deConvLayer(layer, 256) #16
            layer = self.deConvLayer(layer, 128) #32
            layer = self.deConvLayer(layer, 64) #64
            layer = self.deConvLayer(layer, 16) #128
            layer = self.deConvLayer(layer, 3, is_last_layer = True) #256
            model = Model(inputs = inputx, outputs = layer)
            return model
        
        def encode(self, input, training = True):
            mean, logvar = tf.split(self.encoder(input, training = training), num_or_size_splits = 2, axis = 1)
            return mean, logvar
    
        def reparameterize(self, mean, logvar):
            #eps = tf.random.normal(shape = mean.shape)
            #print(f"mean, logvar types: {type(mean), type(logvar)}")
            eps = tf.random.normal(shape = mean.shape, dtype = tf.dtypes.float32)
            return eps * tf.exp(logvar * np.float32(.5)) + mean
            #return eps * tf.exp(logvar * .5) + mean
        
        def decode(self, input, training = True, apply_sigmoid = False):
            logits = self.decoder(input, training = training)
            return logits   
        
        def convLayer(self, input, filter_size):
            #padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            # layer = tf.pad(input, padding, "CONSTANT")
            layer = input
            layer = Conv2D(filters = filter_size, kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            # layer = tf.pad(layer, padding, "CONSTANT")
            layer = Conv2D(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            return layer

        def deConvLayer(self, input, filter_size, is_last_layer = False):
            # padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            # layer = tf.pad(input, padding, "CONSTANT")
            layer = input
            layer = Conv2DTranspose(filters = filter_size, kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            # layer = tf.pad(layer, padding, "CONSTANT")
            if (is_last_layer == True):
                layer = Conv2DTranspose(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same')(layer)
                #layer = keras.activations.tanh(layer)
                layer = keras.layers.Activation('tanh', dtype = 'float32')(layer)
                # layer = BatchNormalization()(layer)
                return layer
            layer = Conv2DTranspose(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same', use_bias = False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            return layer
        
    class AnoVAEGAN:
        def __init__(self, input_shape, custom_bottleneck_size = None):
            self.input_shape = input_shape
            self.custom_bottleneck_size = custom_bottleneck_size
            #self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE, label_smoothing = 0.1)
            self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            self.mse_loss = keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)
            self.learning_rate = 1e-4
            self.generator_optimizer = keras.optimizers.Adam(self.learning_rate, epsilon=1e-04)
            # self.generator_optimizer = keras.optimizers.SGD(self.learning_rate)
            self.generator_optimizer = mixed_precision.LossScaleOptimizer(self.generator_optimizer)
            self.discriminator_optimizer = keras.optimizers.Adam(self.learning_rate, epsilon=1e-04)
            # self.discriminator_optimizer = keras.optimizers.SGD(self.learning_rate)
            self.discriminator_optimizer = mixed_precision.LossScaleOptimizer(self.discriminator_optimizer)
            # self.encoder_optimizer = keras.optimizers.Adam(self.learning_rate)
            # self.encoder_optimizer = mixed_precision.LossScaleOptimizer(self.encoder_optimizer)
            self.batch_size = 16
            self.epochs = 100
            self.loss_weight = 0.2
            self.createNetwork()
            self.model = CVAE(input_shape)
            self.checkpoint_dir = './training_checkpoints'
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer, 
                                                discriminator_optimizer = self.discriminator_optimizer,
                                                # encoder_optimizer = self.encoder_optimizer, 
                                                encoder = self.model.encoder,
                                                decoder = self.model.decoder, 
                                                discriminator = self.discriminator)
        
        def convLayer(self, input, filter_size):
            padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            # layer = tf.pad(input, padding, "CONSTANT")
            layer = input
            layer = Conv2D(filters = filter_size, kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            # layer = tf.pad(layer, padding, "CONSTANT")
            layer = Conv2D(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            return layer

        def deConvLayer(self, input, filter_size, is_last_layer = False):
            # padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            # layer = tf.pad(input, padding, "CONSTANT")
            layer = input
            layer = Conv2DTranspose(filters = filter_size, kernel_size = 3, strides = 1, padding = 'same', use_bias=False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            # layer = tf.pad(layer, padding, "CONSTANT")
            if (is_last_layer == True):
                layer = Conv2DTranspose(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same')(layer)
                #layer = keras.activations.tanh(layer)
                layer = keras.layers.Activation('tanh', dtype = 'float32')(layer)
                # layer = BatchNormalization()(layer)
                return layer
            layer = Conv2DTranspose(filters = filter_size, kernel_size = 4, strides = 2, padding = 'same', use_bias = False)(layer)
            layer = LeakyReLU()(layer)
            layer = BatchNormalization()(layer)
            return layer
        
        def createGenerator(self, input_shape, custom_bottleneck_size = None):
            input = Input(shape = input_shape)
            layer = self.convLayer(input, 64) #128
            layer = self.convLayer(layer, 64) #64
            layer = self.convLayer(layer, 128) #32
            layer = self.convLayer(layer, 256) #16
            layer = self.convLayer(layer, 512) #8
            layer = self.convLayer(layer, 512) #4
            # layer = Flatten()(layer)
            if (custom_bottleneck_size == None):
                layer = Conv2D(filters = 8000, kernel_size = layer.shape[1], strides = layer.shape[2], padding = 'same', name = 'intermediate_conv')(layer)
            
            else:
                layer = Conv2D(filters = custom_bottleneck_size, kernel_size = layer.shape[1], strides = layer.shape[2], padding = 'same', name = 'intermediate_conv')(layer)
            layer = LeakyReLU()(layer)
            # layer = Reshape((4, 4, 500))(layer)
            layer = self.deConvLayer(layer, 512) #4
            layer = self.deConvLayer(layer, 512) #8
            layer = self.deConvLayer(layer, 256) #16
            layer = self.deConvLayer(layer, 128) #32
            layer = self.deConvLayer(layer, 64) #64
            layer = self.deConvLayer(layer, 64) #128
            #layer = self.deConvLayer(layer, 64) #128
            layer = self.deConvLayer(layer, 3, is_last_layer = True)
            model = Model(inputs = input, outputs = layer)
            return model

        def createDiscriminator(self, input_shape):
            vgg_model = VGG19(include_top = False, input_shape = input_shape)
            vgg_model.trainable = False
            model = Sequential()
            for layer in vgg_model.layers:
                layer.trainable = False
                if ('conv' in layer.name):
                    model.add(layer)
                    model.add(BatchNormalization())
                    model.add(Dropout(0.2))
                else:
                    model.add(layer)
            model.add(Flatten())
            model.add(Dense(1, dtype = 'float32'))
            # model.add(Activation('sigmoid', dtype = 'float32'))
            return model
        
        def discriminatorLoss(self, real_output, fake_output):
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            # real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = real_output, labels = tf.ones_like(real_output))
            real_loss = tf.reduce_sum(real_loss) * (1. / self.batch_size)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            # fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_output, labels = tf.zeros_like(fake_output))
            fake_loss = tf.reduce_sum(fake_loss) * (1. / self.batch_size)
            total_loss = real_loss + fake_loss
            return tf.reduce_mean(total_loss)
        
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            #print("TYPES:")
            #print(type(sample))
            #print(type(mean))
            #print(type(logvar))
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        
        
        #def generatorLoss(self, fake_output, images, generated_images, mean, logvar, bt_neck):    
        #    mse_loss = tf.keras.losses.MSE(images, generated_images)
        #    return mse_loss
        
        # def GaussianKLD(self, mu1, lv1, mu2, lv2):
        #     v1 = tf.exp(lv1)
        #     v2 = tf.exp(lv2)
        #     mu_diff_sq = tf.square(mu1 - mu2)
        #     dimwise_kld = .5 * ((lv2 - lv1) + tf.divide(v1, v2) + tf.divide(mu_diff_sq, v2) - 1.)
        #     return tf.reduce_sum(dimwise_kld)
        
        # def generatorLoss(self, fake_output, images, generated_images, mean, logvar):
        #     #1. Discriminator Loss
        #     loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_output, tf.ones_like(fake_output)))
        #     # print(f"Loss D: {loss_D}")
        #     #2. KLD Loss
        #     # loss_KL = tf.reduce_mean(tf.reduce_mean(.5 * ((0 - logvar) + tf.divide(tf.exp(logvar), tf.exp(0)) + tf.divide(tf.square(mean), tf.exp(0)) - 1.), -1.))
        #     loss_KL = self.GaussianKLD(mean, logvar, tf.zeros_like(mean), tf.zeros_like(logvar))/(images.shape[1] * images.shape[2] * images.shape[3])
        #     #3. MSE Loss
        #     loss_mse = tf.reduce_mean(tf.square(images - generated_images))
        #     # print(f"Loss MSE: {loss_mse}")
        #     overall_loss = loss_D*self.loss_weight + loss_mse*(1 - self.loss_weight)
            
        #     return overall_loss, loss_KL 
        
        # def generatorLoss(self, fake_output, images, generated_images, mean, logvar, bt_neck): #LATEST LOSS FUNCTION
        #     ce_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        #     ce_loss = tf.reduce_sum(ce_loss) * (1. / self.batch_size)
            
        #     images = tf.expand_dims(images, axis = -1)
        #     generated_images = tf.expand_dims(generated_images, axis = -1)
            
        #     mse_loss = K.mean(math_ops.squared_difference(generated_images, images))
            
        #     return self.loss_weight*ce_loss + (1 - self.loss_weight)*mse_loss, ce_loss, mse_loss, 0
        
        def generatorLoss(self, fake_output, images, generated_images, mean, logvar, bt_neck, epoch): #SIMPLE LOSS
            ce_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            ce_loss = tf.reduce_sum(ce_loss) * (1. / self.batch_size)
            
            images = tf.expand_dims(images, axis = -1)
            generated_images = tf.expand_dims(generated_images, axis = -1)
            
            mse_loss = self.mse_loss(images, generated_images)
            mse_loss = tf.reduce_sum(mse_loss)
            mse_loss /= tf.cast(tf.reduce_prod(tf.shape(images)), tf.float32)
            
            #if (epoch < 3):
            #    print("0.2 epoch")
            #    return 0.2*ce_loss + (0.8)*mse_loss, tf.reduce_mean(ce_loss), tf.reduce_mean(mse_loss), 0
            #elif (epoch < 10):
            #    return 0.4*ce_loss + (0.6)*mse_loss, tf.reduce_mean(ce_loss), tf.reduce_mean(mse_loss), 0
            #    print("0.4 epoch")
            #elif (epoch < 15):
            #    return 0.6*ce_loss + (0.4)*mse_loss, tf.reduce_mean(ce_loss), tf.reduce_mean(mse_loss), 0
            #    print("0.6 epoch")    
            
            return tf.cast(self.loss_weight*ce_loss + (1 - self.loss_weight)*mse_loss, tf.float32), tf.reduce_mean(ce_loss), tf.reduce_mean(mse_loss), 0
            
        
        # def generatorLoss(self, fake_output, images, generated_images, mean, logvar, bt_neck): #LAST LOSS FUNCTION
            
        #     ce_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        #     # ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_output, labels = tf.ones_like(fake_output))
        #     ce_loss = tf.reduce_sum(ce_loss) * (1. / self.batch_size)
        #     # ce_loss = 0
            
        #     images = tf.expand_dims(images, axis = -1)
        #     generated_images = tf.expand_dims(generated_images, axis = -1)
            
        #     mse_loss = self.mse_loss(images, generated_images)
        #     # print("mse_loss shape: ", mse_loss.shape)
        #     # print("images shape: ", images.shape)
        #     # mse_loss = tf.reduce_sum(mse_loss) * (1. /(self.batch_size))
        #     # mse_loss = tf.nn.compute_average_loss(mse_loss, self.batch_size)
        #     # mse_loss = tf.reduce_sum(mse_loss, axis=0)
        #     mse_loss = tf.reduce_sum(mse_loss)
        #     mse_before = mse_loss
        #     mse_loss /= tf.cast(tf.reduce_prod(tf.shape(images)[1:]), tf.float32)
        #     # mse_loss = tf.reduce_sum(mse_loss)
        #     print("ce loss shape: ", ce_loss.shape)
        #     print("mse loss shape: ", mse_loss.shape)
        #     # return ce_loss
        #     return self.loss_weight*ce_loss + (1 - self.loss_weight)*mse_loss, ce_loss, mse_loss, mse_before
        #     # return mse_loss
        
        # def generatorLoss(self, fake_output, images, generated_images, mean, logvar, bt_neck):
        #     # mean = tf.cast(mean, dtype= tf.float32)
        #     # logvar = tf.cast(logvar, dtype= tf.float32)
        #     # bt_neck = tf.cast(bt_neck, dtype = tf.float32)
        #     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_images, labels=images)
        #     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        #     #print(type(bt_neck))
        #     #logpz = self.log_normal_pdf(bt_neck, 0., 0.)
        #     #logqz_x = self.log_normal_pdf(bt_neck, mean, logvar)
            
        #     log2pi = tf.math.log(2. * np.pi)
        #     logpz = tf.reduce_sum(-.5 * ((bt_neck - 0.) ** 2. * tf.exp(0.) + 0. + log2pi), axis = 1)
        #     logqz_x = tf.reduce_sum(-.5 * ((bt_neck - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = 1)
            
        #     mse_loss = tf.reduce_mean(tf.square(images - generated_images))
            
        #     return (-tf.reduce_mean(logpx_z + logpz - logqz_x))*self.loss_weight + mse_loss*(1-self.loss_weight)
        
        
        #def generatorLoss(self, fake_output, images, generated_images, mean, logvar):
        #    ce_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        #    #kl = - 0.5 * K.sum(1 + logvar - K.square(mean) - K.exp(logvar), axis=-1)
        #    #print("loss shape:")
        #    #print(ce_loss.shape)
        #    #print(kl.shape)
        #    ce_loss = tf.reduce_sum(ce_loss) * (1. / self.batch_size)
        #    #ce_loss = tf.reduce_sum(ce_loss + kl) * (1. / self.batch_size)
        #    #mse_loss = tf.keras.losses.MSE(tf.cast(images, dtype=tf.float16), tf.cast(generated_images, dtype=tf.float16))
        #    mse_loss = tf.keras.losses.MSE(images, generated_images)
        #    
        #    # ce_loss = K.mean(ce_loss + kl)
        #    
        #    #print(ce_loss.shape)
        #    
        #    total_loss = self.loss_weight*ce_loss + (1 - self.loss_weight)*mse_loss
        #    return total_loss 

        def createNetwork(self):
            self.generator = self.createGenerator(self.input_shape, self.custom_bottleneck_size)
            self.discriminator = self.createDiscriminator(self.input_shape)

        @tf.function  
        def train_step(self, images, epoch):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # generated_images = self.generator(images, training = True)
                mean, logvar = self.model.encode(images)
                # mean_copy = tf.identity(mean)
                # logvar_copy = tf.identity(logvar)
                bt_neck = self.model.reparameterize(mean, logvar)
                #print(f"bt_neck type: {type(bt_neck)}")
                # bt_neck_copy = tf.identity(bt_neck)
                generated_images = self.model.decode(bt_neck)
                
                real_output = self.discriminator(images, training = True)
                fake_output = self.discriminator(generated_images, training = True)
                
                # gen_loss, encoder_loss = self.generatorLoss(fake_output, images, generated_images, mean, logvar)
                gen_loss, ce_loss, mse_loss, mse_before = self.generatorLoss(fake_output, images, generated_images, mean, logvar, bt_neck, epoch)
                #gen_loss = self.generatorLoss(fake_output, images, generated_images, mean, logvar, z)
                disc_loss = self.discriminatorLoss(real_output, fake_output)
                
                gen_loss_ori = gen_loss
                # encoder_loss_ori = encoder_loss
                disc_loss_ori = disc_loss
                
                gen_loss = self.generator_optimizer.get_scaled_loss(gen_loss)
                disc_loss = self.discriminator_optimizer.get_scaled_loss(disc_loss)
                # encoder_loss = self.encoder_optimizer.get_scaled_loss(encoder_loss)
                
            gradients_of_generator = gen_tape.gradient(gen_loss, self.model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            # gradients_of_encoder = encoder_tape.gradient(encoder_loss, self.model.encoder.trainable_variables)
            
            gradients_of_generator = self.generator_optimizer.get_unscaled_gradients(gradients_of_generator)
            gradients_of_discriminator = self.discriminator_optimizer.get_unscaled_gradients(gradients_of_discriminator)
            # gradients_of_encoder = self.encoder_optimizer.get_unscaled_gradients(gradients_of_encoder)
            
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            # self.encoder_optimizer.apply_gradients(zip(gradients_of_encoder, self.model.encoder.trainable_variables))
            return gen_loss_ori, disc_loss_ori, ce_loss, mse_loss, mse_before
            
        """
        CHANGES:
            1. Reduce prints as much as possible for faster implementation. #DONE
            2. Save images at max 5 time per epoch.
            3. Save the model on the same name.
            4. Save numpy with model epoch, batch information.
            5. Write new generate_and_save_images()
        """
        def train(self, dataset, epochs, test_data):
            train_batch_count = dataset.__len__()
            print("No. of epochs: ", epochs, flush = True)
            print("  No. of training batches: ", train_batch_count, flush = True)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}", flush = True)
                start = time.time()
                for i in range(train_batch_count):
                    gen_loss_values, disc_loss_values, ce_loss, mse_loss, mse_before = self.train_step(dataset.__getitem__(i), epoch)
                    if (i % 1000 == 0):
                        print(f"Batch {i+1} Completed.")
                        # print(gen_loss_values, disc_loss_values, ce_loss, mse_loss, mse_before)
                        print(f"Generated loss:     {gen_loss_values}")
                        print(f"Discriminator Loss: {disc_loss_values}")
                        print(f"Cross Entropy Loss: {ce_loss}")
                        print(f"MSE Loss:           {mse_loss}")
                        print(f"MSE Before:         {mse_before}")
                        # print("")
                        
                        # pred_image, act_image = self.generate_and_save_images(self.model, 
                        #                                               epoch + 1 + 200, 
                        #                                               test_data.__getitem__(0))
                        
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar("Batch Generator Loss", np.mean(gen_loss_values), step = epoch*train_batch_count + i)
                        tf.summary.scalar("Batch Discriminator Loss", np.mean(disc_loss_values), step = epoch*train_batch_count + i)
                        # tf.summary.scalar("Batch Encoder Loss", np.mean(encoder_loss_values), step = epoch*train_batch_count + i)
                        self.train_summary_writer.flush()
                        # tf.summary.image("Batch Predicted Image", np.expand_dims(pred_image, axis = 0), step = epoch*train_batch_count + i)
                        # tf.summary.image("Batch Actual Image", np.expand_dims(act_image, axis = 0), step = epoch*train_batch_count + i)
                        
                    # if (i %100 == 0):
                    #     pred_image, act_image = self.generate_and_save_images(self.model, epoch + 1, test_data.__getitem__(0))
                        
                pred_image, act_image = self.generate_and_save_images(self.model, 
                                                                      epoch + 1, 
                                                                      test_data.__getitem__(0))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("Epoch Generator Loss", np.mean(gen_loss_values), step = epoch*train_batch_count + i)
                    tf.summary.scalar("Epoch Discriminator Loss", np.mean(disc_loss_values), step = epoch*train_batch_count + i)
                    # tf.summary.scalar("Epoch Encoder Loss", np.mean(encoder_loss_values), step = epoch*train_batch_count + i)
                    self.train_summary_writer.flush()
                    
                self.checkpoint.save(file_prefix = self.checkpoint_prefix + str(epoch + 1))
                print(f"Epoch Completed in: {time.time() - start} secs.")
                print(" ")

        def generate_and_save_images(self, model, epoch, test_input):
            # predictions = model(test_input, training = False)
            mean, logvar = self.model.encode(test_input, training = False)
            z = self.model.reparameterize(mean, logvar)
            predictions = self.model.decode(z, training = False)
            
            with self.train_summary_writer.as_default():
                tf.summary.image("Generated Images", predictions*127.5+127.5, step = epoch)
                tf.summary.image("Actual Image: ", predictions*127.5+127.5, step = epoch)
                self.train_summary_writer.flush()
            
            for i in range(predictions.shape[0]):
                pred_image = predictions[i]*127.5 + 127.5
                act_image = test_input[i]*127.5 + 127.5
                image = np.concatenate([act_image, pred_image], axis = 1)
                cv.imwrite("./output/images4/image_{:03d}_{:02d}.png".format(epoch, i), image)
                
            return predictions, test_input

        # def train(self, dataset, epochs, test_data):
        #     test_batch_count = test_data.__len__()
        #     j=0
        #     for epoch in range(epochs):
        #         start = time.time()
        #         train_batch_count = dataset.__len__()
        #         print("No. of training Batches: ", train_batch_count)
        #         print("No. of testing Batches: ", test_batch_count)
        #         print("Epoch ", epoch+1)
        #         for i in range(train_batch_count):
        #             print("     Batch ", i+1)
        #             start = time.time()
        #             gen_loss_values, disc_loss_values = self.train_step(dataset.__getitem__(i))
        #             print("         Batch training time: ", time.time() - start)
        #             print("         Generator Loss: ", np.mean(gen_loss_values))
        #             print("         Discriminator Loss: ", np.mean(disc_loss_values))
                    

        #             # if (j<test_batch_count):
        #             pred_image, act_image = self.generate_and_save_images(self.generator, 
        #                                         epoch + 1,
        #                                         i, 
        #                                         test_data.__getitem__(0))
        #                 # if (j == test_batch_count - 1 or j == test_batch_count):
        #                 # j = 0
        #             # j += 1
                    
        #             with self.train_summary_writer.as_default():
        #                 tf.summary.scalar("Batch Generator Loss", np.mean(gen_loss_values), step = epoch*train_batch_count + i)
        #                 tf.summary.scalar("Batch Discriminator Loss", np.mean(disc_loss_values), step = epoch*train_batch_count + i)
        #                 tf.summary.image("Batch Predicted Image", np.expand_dims(pred_image, axis = 0), step = epoch*train_batch_count + i)
        #                 tf.summary.image("Batch Actual Image", np.expand_dims(act_image, axis = 0), step = epoch*train_batch_count + i)

        #         if (epoch + 1) %1 == 0:
        #             self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                    
        #         print ("Time for epoch {} is {} secs".format(epoch + 1, time.time() - start))
                
        #     self.generate_and_save_images(self.generator, 100, 100, test_data.__getitem__(1))
            
        # def generate_and_save_images(self, model, epoch, batch, test_input):
        #     predictions = model(test_input, training=False)
            
        #     # for i in range(predictions.shape[0]):
        #     i = random.randrange(0, predictions.shape[0])
        #     pred_image = predictions[i] * 127.5 + 127.5
        #     act_image = test_input[i] * 127.5 + 127.5
        #     image = np.concatenate([act_image, pred_image], axis=1)
        #     np.save("./output/numpys/image_at_epoch_{:04d}_{:03d}.npy".format(epoch, batch), image)
        #     # image = image/255
        #     cv.imwrite("./output/images/image_at_epoch_{:04d}_{:03d}.png".format(epoch, batch), image)
        #     # plt.imshow(image)
        #     # plt.savefig("./output/images/image_at_epoch_{:04d}_{:03d}.png".format(epoch, batch))
        #     # plt.close()
            
        #     print("Image for epoch ", epoch, " batch", batch, " generated.")
        #     return pred_image/255, act_image/255
                
        def change_params(self, input_shape = None, 
                        custom_bottleneck_size = None, 
                        generator_optimizer = None, 
                        discriminator_optimizer = None, 
                        batch_size = None,
                        epochs = None, 
                        loss_weight = None, 
                        checkpoint_dir = None, 
                        learning_rate = None,
                        logs = None,
                        ):
            if (input_shape != None):
                self.input_shape = input_shape
            if (custom_bottleneck_size != None):
                self.custom_bottleneck_size = custom_bottleneck_size
            if (generator_optimizer != None):
                self.generator_optimizer = generator_optimizer
            if (discriminator_optimizer != None):
                self.discriminator_optimizer = discriminator_optimizer
            if (batch_size != None):
                self.batch_size = batch_size
            if (epochs != None):
                self.epochs = epochs
            if (loss_weight != None):
                self.loss_weight = loss_weight
            if (checkpoint_dir != None):
                self.checkpoint_dir = checkpoint_dir
                self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                self.checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer, 
                                                      discriminator_optimizer = self.discriminator_optimizer, 
                                                    #   encoder_optimizer = self.encoder_optimizer,
                                                      encoder = self.model.encoder, 
                                                      decoder = self.model.decoder, 
                                                      discriminator = self.discriminator)
            if (learning_rate != None):
                self.learning_rate = learning_rate
                self.generator_optimizer =  keras.optimizers.Adam(self.learning_rate)
                self.discriminator_optimizer =  keras.optimizers.Adam(self.learning_rate)
            if (logs != None):
                self.log_path = os.path.join(logs, "LOG_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                self.train_summary_writer = tf.summary.create_file_writer(self.log_path)
                
            return self.input_shape, self.custom_bottleneck_size, self.generator_optimizer, self.discriminator_optimizer, self.batch_size, self.epochs, self.loss_weight, self.checkpoint_dir, self.learning_rate, self.log_path, self.checkpoint_prefix
                
        def load_model_checkpoint(self, checkpoint_path):
            self.checkpoint.restore(checkpoint_path)
        
        def printModelSummary(self):
            print("------------------------------------------------GENERATOR NETWORK------------------------------------------------")
            print(self.generator.summary())
            print("------------------------------------------------DISCRIMINATOR NETWORK------------------------------------------------")
            print(self.discriminator.summary())
            


