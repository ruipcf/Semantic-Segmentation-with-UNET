import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from random import randint
import random
import cv2


class SemanticSegmentation():
    def __init__(self, epochs, batch_size, loss_function, metrics, learning_rate, trainable, model_name):
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.model_name = model_name

        self.IMG_SHAPE = 256

        if self.trainable:
            self.modelsPath = "models"
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.modelFileName = "unet" + "E" + str(self.epochs) + "_LR" + str(
                self.learning_rate) + "_" + self.timestamp + ".hdf5"

        # dataset manipulation
        self.images = self.load_dataset("images")
        self.masks = self.load_dataset("labels")

        aux = list(zip(self.images, self.masks))
        random.shuffle(aux)
        self.images, self.masks = zip(*aux)

        self.classes = self.load_classes()

        # split into train & validation
        split = int(0.8 * len(self.images))
        self.train_images = self.images[:split]
        self.val_images = self.images[split:]
        self.train_masks = self.masks[:split]
        self.val_masks = self.masks[split:]

        self.train_generator = self.custom_generator(self.train_images, self.train_masks, aug = True)
        self.val_generator = self.custom_generator(self.val_images, self.val_masks, aug = False)

        # run model and metrics
        if self.trainable:
            self.run_model()
            self.model_metrics()
        else:
            # comment to use a model pre-trainned
            # uncoment to retrain the model
            #self.run_model(self.model_name)

            self.model_metrics()

    def load_dataset(self, folder):
        list = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                list.append(img)

            # load only 100 images
            if len(list) == 100:
                break

        return list

    def load_classes(self):
        # https://bitbucket.org/visinf/projects-2016-playing-for-data/src/master/label/initLabels.m

        classes = {'unlabeled':[0,0,0], 'ego vehicle':[0,0,0], 'rectification border':[0,0,0], 'out of roi':[0,0,0], 'static':[20,20,20],
        'dynamic':[111,74,0], 'ground':[81,0,81], 'road':[128,64,128], 'sidewalk':[244,35,232], 'parking':[250,170,160],
        'rail track':[230,150,140], 'building':[70,70,70], 'wall':[102,102,156], 'fence':[190,153,153], 'guard rail':[180,165,180],
        'bridge':[150,100,100], 'tunnel':[120,120,90], 'pole':[153,153,153], 'polegroup':[153,153,153], 'traffic light':[250,170,30],
        'traffic sign':[220,220,0], 'vegetation':[107,142,35], 'terrain':[152,251,152], 'sky':[70,130,180], 'person':[220,20,60],
        'rider':[255,0,0], 'car':[0,0,142], 'truck':[0,0,72], 'bus':[0,60,100], 'caravan':[0,0,90],
        'trailer':[0,0,110], 'train':[0,80,100], 'motorcycle':[0,0,230], 'bicycle':[19,11,32], 'license plate':[0,0,142]}

        return classes

    def augmentation(self, image, mask, aug):
        image = cv2.resize(image, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)
        image = image/255
        mask = cv2.resize(mask, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)

        batchMasks = np.zeros(((self.IMG_SHAPE, self.IMG_SHAPE) + (len(self.classes),)))
        for classe in self.classes:
            idx = list(self.classes.keys()).index(classe)
            pixels = self.classes.get(classe)
            pixels = list(reversed(pixels))
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if list(mask[i][j]) == pixels:
                        batchMasks[i,j,idx] = 1
            #cv2.imshow(classe, batchMasks[:,:,idx])
            #cv2.waitKey()

        if aug:
            op = 0
            # randomly choose between flip horizontal
            op = randint(0, 1)
            if op:
                image = cv2.flip(image, 1)
                for i in range(len(self.classes)):
                    batchMasks[:,:,i] = cv2.flip(batchMasks[:,:,i], 1)

            # randomly choose translation
            op = randint(0, 1)
            if op:
                value = random.uniform(0, 1)
                ratio = random.uniform(-value, value)
                h, w = image.shape[:2]
                to_shift = w * ratio
                if ratio > 0:
                    image = image[:, :int(w - to_shift), :]
                    for i in range(len(self.classes)):
                        aux = batchMasks[:, :, i]
                        aux = aux[:, :int(w - to_shift)]
                        batchMasks[:, :, i] = cv2.resize(aux, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)
                if ratio < 0:
                    image = image[:, int(-1 * to_shift):, :]
                    for i in range(len(self.classes)):
                        aux = batchMasks[:, :, i]
                        aux = aux[:, int(-1 * to_shift)]
                        batchMasks[:, :, i] = cv2.resize(aux, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)

                image = cv2.resize(image, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)

            # randomly choose zoom ratio
            op = randint(0, 1)
            if op:
                value = random.uniform(0.3, 1)
                taken = int(value * self.IMG_SHAPE)
                h_start = random.randint(0, self.IMG_SHAPE - taken)
                w_start = random.randint(0, self.IMG_SHAPE - taken)
                image = image[h_start:h_start + taken, w_start:w_start + taken, :]
                image = cv2.resize(image, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)

                for i in range(len(self.classes)):
                    aux = batchMasks[:, :, i]
                    aux = aux[h_start:h_start + taken, w_start:w_start + taken]
                    batchMasks[:, :, i] = cv2.resize(aux, (self.IMG_SHAPE, self.IMG_SHAPE), cv2.INTER_CUBIC)

        return image, batchMasks

    def custom_generator(self, x, y, aug = True):
        while 1:
            nBatches = int(np.ceil(len(x) / self.batch_size))
            for batchID in range(nBatches):
                images = np.zeros(((self.batch_size,) + (self.IMG_SHAPE,self.IMG_SHAPE) + (3,)))
                masks  = np.zeros(((self.batch_size,) + (self.IMG_SHAPE,self.IMG_SHAPE) + (len(self.classes),)))

                idxBatch = 0
                while idxBatch < self.batch_size:

                    image, batchMasks = self.augmentation(x[idxBatch],y[idxBatch], aug)

                    images[idxBatch, :, :, :] = image
                    masks[idxBatch, :, :, :] = batchMasks

                    idxBatch = idxBatch + 1

                yield images, masks

    def run_model(self, pretrained_weights = None):

        do_batch_normalization = True
        use_transpose_convolution = False

        inputs = tf.keras.layers.Input((self.IMG_SHAPE,self.IMG_SHAPE,3))
        conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
        if do_batch_normalization:
            conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation('relu')(conv1)
        conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
        if do_batch_normalization:
            conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation('relu')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
        if do_batch_normalization:
            conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)
        conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
        if do_batch_normalization:
            conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
        if do_batch_normalization:
            conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)
        conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
        if do_batch_normalization:
            conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
        if do_batch_normalization:
            conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Activation('relu')(conv4)
        conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
        if do_batch_normalization:
            conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Activation('relu')(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
        if do_batch_normalization:
            conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Activation('relu')(conv5)
        conv5 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
        if do_batch_normalization:
            conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Activation('relu')(conv5)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)

        if use_transpose_convolution:
            up6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(drop5)
        else:
            up6 = tf.keras.layers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        if do_batch_normalization:
            up6 = tf.keras.layers.BatchNormalization()(up6)
        up6 = tf.keras.layers.Activation('relu')(up6)
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
        if do_batch_normalization:
            conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Activation('relu')(conv6)
        conv6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
        if do_batch_normalization:
            conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Activation('relu')(conv6)

        if use_transpose_convolution:
            up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(conv6)
        else:
            up7 = tf.keras.layers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        if do_batch_normalization:
            up7 = tf.keras.layers.BatchNormalization()(up7)
        up7 = tf.keras.layers.Activation('relu')(up7)
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
        if do_batch_normalization:
            conv7 = tf.keras.layers.BatchNormalization()(conv7)
        conv7 = tf.keras.layers.Activation('relu')(conv7)
        conv7 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
        if do_batch_normalization:
            conv7 = tf.keras.layers.BatchNormalization()(conv7)
        conv7 = tf.keras.layers.Activation('relu')(conv7)

        if use_transpose_convolution:
            up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(conv7)
        else:
            up8 = tf.keras.layers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        if do_batch_normalization:
            up8 = tf.keras.layers.BatchNormalization()(up8)
        up8 = tf.keras.layers.Activation('relu')(up8)
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
        if do_batch_normalization:
            conv8 = tf.keras.layers.BatchNormalization()(conv8)
        conv8 = tf.keras.layers.Activation('relu')(conv8)
        conv8 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
        if do_batch_normalization:
            conv8 = tf.keras.layers.BatchNormalization()(conv8)
        conv8 = tf.keras.layers.Activation('relu')(conv8)

        if use_transpose_convolution:
            up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
        else:
            up9 = tf.keras.layers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        if do_batch_normalization:
            up9 = tf.keras.layers.BatchNormalization()(up9)
        up9 = tf.keras.layers.Activation('relu')(up9)
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
        if do_batch_normalization:
            conv9 = tf.keras.layers.BatchNormalization()(conv9)
        conv9 = tf.keras.layers.Activation('relu')(conv9)
        conv9 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
        if do_batch_normalization:
            conv9 = tf.keras.layers.BatchNormalization()(conv9)
        conv9 = tf.keras.layers.Activation('relu')(conv9)
        conv10 = tf.keras.layers.Conv2D(len(self.classes), 1, activation='softmax', kernel_initializer='he_normal')(conv9)

        model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

        if pretrained_weights:
            model.load_weights(pretrained_weights)
            modelFilePath = pretrained_weights
        else:
            modelFilePath = os.path.join(self.modelsPath, self.modelFileName)

        model.summary()

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        learningrate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.compile(
            optimizer=Adam(self.learning_rate),
            loss=self.loss_function,
            metrics=[self.metrics])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(modelFilePath,
                                                                 monitor='val_loss',
                                                                 verbose=1,
                                                                 save_best_only=True)

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        Ntrain = len(self.train_images)
        self.epochs_steps = np.ceil(Ntrain / self.batch_size)
        Nval = len(self.val_images)
        self.validation_steps = np.ceil(Nval / self.batch_size)

        history = model.fit(
            self.train_generator,
            steps_per_epoch=self.epochs_steps,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=[checkpoint_callback, tensorboard_callback, earlystopping_callback, learningrate_callback]
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        #plt.show()

    def model_metrics(self):
        if self.trainable:
            modelFilePath = os.path.join(self.modelsPath, self.modelFileName)
        else:
            modelFilePath = model_name
            Nval = len(self.val_images)
            self.validation_steps = np.ceil(Nval / self.batch_size)

        model = tf.keras.models.load_model(modelFilePath)

        results = model.predict(self.val_generator,steps=self.validation_steps, verbose=1)

        count = 0
        batchMasksPredicted = np.zeros(((len(results),) + (self.IMG_SHAPE, self.IMG_SHAPE)))
        for r in results:
            batchMasksPredicted[count] = np.argmax(r,axis= -1)
            count += 1
            #plt.imshow(batchMasksPredicted)
            #plt.show()

        fig, axs = plt.subplots(3, 3)
        fig.suptitle('Examples of model prediction', fontsize=16)
        for i in range(3):
            axs[i, 0].imshow(self.val_images[i])
            axs[i, 1].imshow(self.val_masks[i])
            axs[i, 2].imshow(batchMasksPredicted[i])
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            axs[i, 2].axis('off')
            axs[i, 0].title.set_text('Image')
            axs[i, 1].title.set_text('Mask')
            axs[i, 2].title.set_text('Predicted Mask')
        plt.show()

epochs = 5
batch_size = 2
loss_function = "categorical_crossentropy"
metrics = ["accuracy"]
learning_rate = 1e-4
trainable = False # True for training and False for evaluation
model_name = "models/unetE10_LR0.0001_20220412-153132.hdf5"
model = SemanticSegmentation(epochs,batch_size,loss_function,metrics,learning_rate,trainable,model_name)