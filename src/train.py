import pandas as pd
import numpy as np
import preprocess
import config
import engine
import dataset
import models
from sklearn.model_selection import train_test_split
def run():

    train_data = pd.read_csv(config.DATA_DIR + 'S40AR_train_data.csv')
    action_encoder, train_data = preprocess.encode_target(train_data, 'action', 'action_label')
    action_cls_encoder, train_data = preprocess.encode_target(train_data, 'action_class', 'action_class_label')

    train_val, test = train_test_split(train_data,test_size = 0.15, random_state = 42, stratify = train_data[['action_label','action_class_label']])
    train, valid = train_test_split(train_val, test_size = 0.15, random_state = 42, stratify = train_val[['action_label','action_class_label']])

    # Training params
    train_params = {'dim':config.IMG_DIM,
        'img_size':config.IMG_SIZE,
        'batch_size':config.BATCH_SIZE,
        'data_mean':0.,
        'data_std':255.0,
        'Augment':True,
        'shuffle':True,
        'data_prefix': config.IMAGE_DIR}

    # No shuffle because we don't compute GD here!
    val_params = {'dim':config.IMG_DIM,
        'img_size':config.IMG_SIZE,
        'batch_size':config.BATCH_SIZE,
        'data_mean':0.,
        'data_std':255.0,
        'Augment':False,
        'shuffle':False,
        'data_prefix': config.IMAGE_DIR}

    aug_train_generator = dataset.ImageDataLoader(train,**aug_params) # Training generator augment
    validation_generator = dataset.ImageDataLoader(valid,**val_params) # Validation generator

    inception = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape = config.IMG_DIM)

    inception.trainable = False
    model_histories = {}
    incep_preprocess = tf.keras.applications.inception_v3.preprocess_input

    incep_model = models.ActionModel(input_shape=IMG_DIM,
                                preprocess_input = incep_preprocess,
                                base_model = inception,
                            action_hidden_size_one=1024,
                            action_hidden_size_two=256,
                            action_cls_hidden_size=512)

    model_histories['Inception'] = compile_and_fit(incep_model,
                                            aug_train_generator,
                                            validation_generator,
                                            'models/incep_base',
                                            ckpt_name='incep_base_new.ckpt')
    engine.model_result(incep_model, validation_generator)



    if __name__ == '__main__':
        run()
    

