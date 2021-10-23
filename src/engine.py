# Call back (Early Stopping, Tensorboard and Model Checkpoint)
import numpy as np 
import tensorflow as tf
import config
def get_callbacks(name, ckpt_name):
  return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR,ckpt_name),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True),
    tf.keras.callbacks.TensorBoard(logdir/name)
  ]

def compile_and_fit(model, training_generator, validation_generator, name, ckpt_name, learning_rate = config.LR, max_epochs=config.NUM_EPOCHS):
  
  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate,
    decay_steps=config.STEPS_PER_EPOCH*100,
    decay_rate=1,
    staircase=False)
  # Optimizer
  optimizer = tf.keras.optimizers.Adam(lr_schedule)

  model.compile(optimizer=optimizer,
                loss = {'action':'sparse_categorical_crossentropy',
                    'action_class':'sparse_categorical_crossentropy'},
              metrics={'action':'accuracy',
                       'action_class':'accuracy'})

  model.summary()

  history = model.fit(
      training_generator,
      steps_per_epoch = config.STEPS_PER_EPOCH,
      batch_size = config.BATCH_SIZE,
      epochs=max_epochs,
      validation_data=validation_generator,
      callbacks=get_callbacks(name, ckpt_name),
      verbose=1)

  return history

def model_result(model, generator):
  
  action_preds = []
  action_trues = []
  action_cls_preds = []
  action_cls_trues = []
  N_images = len(generator.image_ids)
  
  batches = 0

  # iterate through the data generator and predict for each batch
  # hold the predictions and labels
  
  for x,[y,z] in generator.__iter__():
    action_pred, action_cls_pred = model.predict(x, verbose=0)
    
    action_pred = np.argmax(action_pred, axis = 1)
    action_cls_pred = np.argmax(action_cls_pred, axis = 1)
    # actions
    action_preds = action_preds + action_pred.tolist()
    action_trues = action_trues + y.tolist()
    
    # action class
    action_cls_preds = action_cls_preds + action_cls_pred.tolist()
    action_cls_trues = action_cls_trues + z.tolist()

    batches += 1
    if batches >= N_images / BATCH_SIZE:
        #break the loop
        break
  print("Action Macro F1-score: ", metrics.f1_score(action_trues, action_preds,average = "macro"))
  print("Action Class Macro F1-Score : ", metrics.f1_score(action_cls_trues,action_cls_preds,average= "macro"))
  print("Action Classification Report \n", metrics.classification_report(action_trues, action_preds, target_names = target_action))
  print("Action Class Classification Report \n", metrics.classification_report(action_cls_trues, action_cls_preds, target_names = target_action_class))