# Human Activity Recognitions as Multi-task learning
<p> Human Activity Recognition is an ongoing research field in computer vision and image information retrieval. While it has myriads of application in surveillance, image analysis, annotations, recognising actions from still images poses many challenges. One of the fundamental reasons is that the same human
actions can vary in postures, viewpoints and complex backgrounds, making it difficult to recognising
patterns. Despite this, several deep learning-based methods have been developed over the years to
overcome such issues, achieving superior performance. We leverage this existing knowledge from the
literature and apply it to our tasks in this work. Here, we are interested in identifying "human actions" and
"action classes" from still images. Since we desire a single network capable of generating two outputs
simultaneously, we designed a multi-model architecture to solve this. We used a modified version of the
Stanford Action 40 image dataset, a popular database actively used to validate novel methods in the
computer vision literature.</p>

## Evaluation Framework

<p> Stanford Action 40 images have been a benchmark dataset to solve action recognition using various
proposed deep learning techniques. The original data consists of 40 different human actions; however, our
version has 21 actions and five action classes. The labels correspond to various activities performed by
humans and group of categories to which it belongs. For example, if a person is riding a horse, the action
is labelled as 'riding a horse' with an action class 'interacting with animal'. Similarly, this applied to entire
images in the data. We used a total of 3030 as training examples and 2100 for testing purposes. Since our
goal is to develop a single neural network capable of identifying both action and action classes, we define
this as a multi-output classification problem where a single input determines two separate outputs. To
evaluate the performance of our classifier, we use the following metrics based on the evidence from
exploratory data analysis:
<li> Macro-average f1-score: Due to the uneven distribution of the target labels, we want to give equal
importance to each 'action' and 'action class'. F1-Score is a harmonic mean of precision and recall that helps
us understand how the classifier can precisely predict accurate labels and the proportion of predicted labels
given the total number of targets. It is achieved by closely monitoring the performance regardless of the
class weights. Although existing works have used the mean average precision(mAP) score to determine the
  performance, we find the macro average f1-score similar and intuitive. </li>
  
<li> Confusion Matrix: Confusion matrix helps to better understand the classifier's performance by looking at
  how well it can recognise actions and action classes over misclassifications. </li>
For a fair comparison between different CNN architectures, we randomly divide the datasets keeping targets
labels at the same distribution into three splits: Training, Validation, and Test with a proportion of 0.7, 0.15, 
0.15. We used training and validation for training the models and tuning hyperparameters, whereas the test
set is kept independent and only used during final model diagnosis judgements. In addition, we also used
out-of-sample testing for error analysis and to see how well the model performs outside of the given set of
images. </p>

## Methods

<p> Deep learning models require a large amount of training data to perform better. However, acquiring labels
can be very expensive and time-consuming. Transfer learning is a machine learning paradigm that has
successfully overcome this challenge. Instead of training a complex neural network from scratch, we
can transfer knowledge from one task to another and obtain outstanding performance even with data. We
leverage pre-trained models developed using ImageNet datasets and transfer that knowledge into our action
recognition task with this motivation. This decision was supported when we observed poor performance
while training a convolution neural network from scratch. </p>

<p> We selected three existing pre-trained models for our experiment: VGG-16, InceptionV3 and
NasNetMobile. VGG-16 is one of the most popular convolution neural networks, which uses multiple
blocks of convolution layers of filter size 3X3 and max-pooling layer size of 2X2. The convolution layer
has a stride of 1 with the padding set to 'same', whereas the stride of max-pooling is 2. This model has also
been considered a strong baseline for action recognition. Our action classifier consisted network of shapes
VGG X 1024 X 256 X 21, and the action class consisted of VGG X 512 X 5. We used 'relu' activation for
the dense layers and softmax to output the probabilities of each class.
InceptionV3 consists of several inception blocks, a combination of convolutions, average and max
pooling, concatenation and dropout layers. The convolution filter size is 7X7 and has improved
performance over VGG in both Imagenet and action recognition tasks. Similar to our former architecture,
this model uses the exact network structure in the top layers.
Neural Architecture Search Network (NasNet) was developed by Google Brain, changing the entire
realm of Image Classification. Instead of explicitly defining Convolutional architectures and training them
with a tremendous volume of data, NasNet transforms this into a reinforcement problem. The main idea of
this model is to sample a child network on a small dataset and search for the best CNN architecture based
on validation performance and then transfer that information into a much large dataset or problem. This
model has shown remarkable performance in image classification as well as more specific tasks like action
recognition. We used a miniature version of this architecture with the network shape NasNet X 512 X 256
X21 for action and NasNet X 512 X 5 for action class, respectively. </p>

## Experiment
<p> Our input images had a variable size, with most images having a width and height of 400X300. Since most
CNN models prefer fixed input size, we resized the image to shape 224 X 224 X 3 and validated it by
plotting them. We noticed that only the resolution of the image decreased but was easily recognisable. We
then developed a custom data loader for generating batches of images for training and validation. As our
training images were limited, we noticed overfitting at the initial experiment for all used techniques. To
overcome this, we augmented our training images using different approaches such as random cropping,
horizontal flip (left to right), random brightness, and rotation, which realistically modified the images into 
different variations of training samples. </p>

<p> Based on multiple observations, we included dropout and weight
regularisers to reduce model complexity. We first froze all the layers from the base model and trained added
fully connected layers with a batch size of 64 for 100 epochs. We used adam as the optimiser with an initial
learning rate of 0.01. To reduce the learning rate gradually, we applied an inverse time decay scheduler
with a decay rate of 1. Once we achieved a better performance of the validation set, we fine-tuned the upper
layers of the base model so that it could capture task-specific features. For this step, we used a lower
learning rate of 5e-5 and 2e-5 to avoid catastrophic forgetting from the pre-trained model and trained for
ten epochs. </p>

## Result

| Methods        | Action Test F1 Score          | Action Class Test F1 Score  |
| ------------- |:-------------:| -----:|
| VGG-16     | 0.615  | 0.811 |
| InceptionV3      | 0.774      |   0.938 |
| NasnetMobile | 0.763      |    0.920 |

![cm](https://github.com/nischaybikramthapa/multi-task-activity-recognition/blob/main/images/nas.JPG)
![cm](https://github.com/nischaybikramthapa/multi-task-activity-recognition/blob/main/images/nas1.JPG)
<p style='text-align': justify> <b> Figures: Results from Nasnet Mobile </b> </p>

![cm](https://github.com/nischaybikramthapa/multi-task-activity-recognition/blob/main/images/incep.JPG)
![cm](https://github.com/nischaybikramthapa/multi-task-activity-recognition/blob/main/images/incep1.JPG)

<p style='text-align': justify> <b> Figures: Results from InceptionV3 </b> </p>
          

## Performance Analysis

<p> We observed InceptionV3 outperforming other models with a macro
average f1 score of 0.774 on identifying action and 0.938 on action class. In addition, we compared model
performance using a confusion matrix, visualising the proportion of misclassifications based on label
distribution and using out of sample images. We noticed a similar performance
from InceptionV3 and Nasnet Mobile; however, InceptionV3 made few errors overall. When looking
closely at errors, there was no clear winner. Both these models made similar errors when classifying actions
that were closely related. For instance, actions like 'texting message' and 'phoning' both included an image
of the phone. </p>

<p> Similarly, actions like 'cooking' and 'cutting vegetables' involved using kitchen appliances which were
difficult for both of these models to distinguish. The reason for this could be the similar human posture
observed in these actions. A severe error to be noted was when both models predicted action and action
classes that were completely different: 'texting message' classified as 'climbing', 'playing guitar' as 'rowing
a boat' and 'playing musical instrument' as 'interacting with animal'. These might be due to the class
imbalances where the model labelled more images as the majority class labels. To overcome these in the
future, we intend to add more images to balance the targets, extract features based on poses, and increase
the penalty for making incorrect predictions on minority labels. </p>

<p> We conclude that InceptionV3 was slightly better based on overall performance, considering fewer errors
on both complex and straightforward action and action classes and finally predictions made on external
images. We used this model to make our final submission and would consider using it in real-world settings
with a future work of improving performance on predicting actions. </p>

# Code
Experiments can be found <a href = "https://github.com/nischaybikramthapa/multi-task-activity-recognition/tree/main/notebooks"> here</a> and code to reproduce is available <a href = "https://github.com/nischaybikramthapa/multi-task-activity-recognition/tree/main/src"> here </here>
