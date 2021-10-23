import matplotlib.pyplot as plt

def plot_from_generator(generator):
  plt.figure(figsize = (20,8))
  i = 0
  for x,[y,z] in generator.__iter__():
    plt.subplot(2,4,i+1)
    plt.imshow(x[0,:,:,0])
    plt.label(str(y[i])+ '_' + str(z[i]))
    plt.colorbar()
    plt.grid(False)
    i += 1
    if i == 7:
      break
  plt.tight_layout(rect = [0, 0.03, 1, 0.95])
  return plt.show()

def plot_learning_curve(train_loss, val_loss, train_metric, val_metric, title, metric_name='Accuracy'):
    plt.figure(figsize=(15,8))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, 'r--')
    plt.plot(val_loss, 'b--')
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(train_metric, 'r--')
    plt.plot(val_metric, 'b--')
    plt.xlabel("epochs")
    plt.ylabel(metric_name)
    plt.legend(['train', 'val'], loc='upper left')
    plt.suptitle(title)
    plt.show()

def plot_both_curves(histories, model_name):
  # Action
  plot_learning_curve(histories[model_name].history['action_loss'],
                      histories[model_name].history['val_action_loss'],
                      histories[model_name].history['action_accuracy'],
                      histories[model_name].history['val_action_accuracy'],
                      "Action Model's Performance")
  # Action Class
  plot_learning_curve(histories[model_name].history['action_class_loss'],
                      histories[model_name].history['val_action_class_loss'],
                      histories[model_name].history['action_class_accuracy'],
                      histories[model_name].history['val_action_class_accuracy'],
                      'Learning Curves')