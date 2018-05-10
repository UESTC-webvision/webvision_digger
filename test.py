import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from teacher import teacher_predict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

num_classes  = 10
batch_size   = 128
epochs       = 200
iterations   = 391
Tempurate    = 20
alpha        = 0.5
weight_decay = 0.0001
log_filepath = r'./student_logs/'

sess = tf.Session()
K.set_session(sess)

if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001

def data_preprocess(x_train, x_test):
    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
    return x_train, x_test

def bulid_model():
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv1', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))
    return model

def my_loss(y_true, y_pred):
    global y_teacher
    soft_target = K.softmax(y_teacher/Tempurate)
    hard_loss = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))
    soft_loss = K.mean(K.categorical_crossentropy(soft_target, y_pred/Tempurate, from_logits=True))
    return hard_loss * (1 - alpha) + soft_loss * alpha * Tempurate * Tempurate

def train_model():

    # data loading
    global y_teacher
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = data_preprocess(x_train, x_test)

    # load pretrained weight from VGG19 by name
    model = bulid_model()
    # model.load_weights(filepath, by_name=True)

    # -------- optimizer setting -------- #
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss=my_loss,
                  metrics=['accuracy'])

    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)

    for epoch in epochs:
        batch = datagen.flow(x_train, y_train, batch_size=batch_size)
        for ite in iterations:
            X, Y = batch[ite][0], batch[ite][1]
            y_teacher = teacher_predict(X)
            model.train_on_batch(X, Y)
        pred = model.predict(x_test, batch_size=128)
        correct_prediction = K.equal(K.argmax(pred, 1), K.argmax(y_test, 1))
        accuracy = K.mean(K.cast_to_floatx(correct_prediction))
        print('Epoch %d : ACC: %g' % (epoch, accuracy))

    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test),
                        verbose=2)
    model.save('student_baseline.h5')
