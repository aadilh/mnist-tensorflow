#MNIST handwritting recognition using TensorFlow

This repository is for the implementation of handwritting recogition with MNIST dataset using TensorFlow.

#Get Started

To run the programs please install TensorFlow with Python 2.7, using the instructions given [here](tensorflow/g3doc/get_started/os_setup.md)

For running basic softmax regression run the following command:
```bash
# Using default configuration for logs directory, batch size and number of iterations:
$ python softmaxReg.py

# For custom configuration use the following command:
$ python softmaxReg.py --max_it <no. of iterations> --batch <batch size> --log_dir <path to logs directory>

# For tensorboard visualization run following command and  visit http://locahost:6006:
$ tensorboard --logdir <path to logs directory>
```
#Data Flow Graphs

##Basic Softmax Implementation

###Overview Graph

![Overview Graph](./images/overview_graph.png)

###Expanded Graph

![Expanded Graph](./images/expanded_graph.png)

#Results

##Basic Softmax Implementation

Following results were obtained with batch size = 100 and number of iterations = 1000

###Accuracy

![Expanded Graph](./images/accuracy_basic.png)

###Cross Entropy

![Expanded Graph](./images/xentropy_basic.png)

###Weights Visualization

####0:

<img src="./images/weight_0.png" width="200" height="200" />


####1:

<img src="./images/weight_1.png" width="200" height="200" />


####2:

<img src="./images/weight_2.png" width="200" height="200" />


####3:

<img src="./images/weight_3.png" width="200" height="200" />


####4:

<img src="./images/weight_4.png" width="200" height="200" />


####5:

<img src="./images/weight_5.png" width="200" height="200" />


####6:

<img src="./images/weight_6.png" width="200" height="200" />


####7:

<img src="./images/weight_7.png" width="200" height="200" />


####8:

<img src="./images/weight_8.png" width="200" height="200" />


####9:

<img src="./images/weight_9.png" width="200" height="200" />
