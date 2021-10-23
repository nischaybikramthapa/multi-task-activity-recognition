import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

class ActionModel:

    def __init__(self, input_shape,
                 preprocess_input,
                 base_model,
                 action_hidden_size_one,
                 action_hidden_size_two,
                 action_cls_hidden_size,
                 action_dropout=0.0,
                 action_l2_prop=0.0,
                 action_cls_dropout=0.0,
                 action_cls_l2_prop=0.0,
                 training=False):
        self.input_shape = input_shape
        self.preprocess_input = preprocess_input
        self.base_model = base_model
        self.action_hidden_size_one = action_hidden_size_one
        self.action_hidden_size_two = action_hidden_size_two
        self.action_cls_hidden_size = action_cls_hidden_size
        self.action_dropout = action_dropout
        self.action_cls_dropout = action_cls_dropout
        self.action_l2_prop = action_l2_prop
        self.action_cls_l2_prop = action_cls_l2_prop
        self.training = training
        self.action = 21
        self.action_cls = 5
        self.assemble_full_model()
    
    def __build_action(self, inputs):
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=self.training)
        x = Flatten()(x)
        x = Dense(self.action_hidden_size_one, kernel_initializer='he_uniform',
                  kernel_regularizer=tf.keras.regularizers.l2(self.action_l2_prop))(x)
        x = Activation('relu')(x)
        x = Dropout(self.action_dropout)(x)
        x = Dense(self.action_hidden_size_two, kernel_initializer='he_uniform',
                  kernel_regularizer=tf.keras.regularizers.l2(self.action_l2_prop))(x)
        x = Activation('relu')(x)
        x = Dropout(self.action_dropout)(x)
        x = Dense(self.action)(x)
        x = Activation('softmax', name="action")(x)
        return x

    def __build_action_class(self, inputs, base_model):
        x = self.preprocess_input(inputs)
        x = base_model(x, training=self.training)
        x = Flatten()(x)
        x = Dense(self.action_cls_hidden_size, kernel_initializer='he_uniform',
                  kernel_regularizer=tf.keras.regularizers.l2(self.action_cls_l2_prop))(x)
        x = Activation('relu')(x)
        x = Dropout(self.action_cls_dropout)(x)
        x = Dense(self.action_clss)(x)
        x = Activation('softmax', name="action_class")(x)
        return x

    def assemble_full_model(self, name="Model"):
        """
        Used to assemble our multi-output model CNN.
        """
        input_layer = Input(shape=self.input_shape)
        action_model = self.__build_action(
            input_layer)
        action_class_model = self.__build_action_class(
            input_layer)
        model = Model(inputs=input_layer, outputs=[action_model, action_class_model],
                      name=name)
        return model
