Handwritten-Digit-Classifier
============================

Handwitten digit classifier based on MNIST dataset (http://yann.lecun.com/exdb/mnist/). MNIST is a dataset of handwritten 
digits with a training set of 60,000 examples and a test set of 10,000 examples.  
It is my current project and I'm trying to improve the results on the test set by trying different machine learning techniques. 

Right now I've implemented a linear multiclass classifier that correctly classifies 89.36% of test set. Learning is done 
using stochastic gradient descent on the multiclass hinge loss function.

Next, I plan to implement K-Nearest Neighbors algorithm and SVMs in the hope to decrease the classification error below 5%.
