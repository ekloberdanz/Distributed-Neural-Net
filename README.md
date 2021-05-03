## Distributed Neural Network Training with MPI

In this project I implement a parallel neural network in C++ leveraging Eigen for linear
algebra operations, and MPI for data parallelism to reduce training time via distributed
learning by spreading the training data across multiple processors.

The neural network I implemented contains two dense layers, a non-linear ReLU
activation function, a softmax activation function, and is optimized via categorical cross
entropy loss and stochastic gradient descent. The training data is divided into partitions
equal to the total number of MPI processes p - 1, which are used as worker nodes. The
neural network model itself is replicated in each of these worker nodes. Therefore, each
worker has a copy of the neural network and operates only on a subset of the training
data. One of the processors in the cluster stores the global model parameters, and acts
as a parameter server that synchronizes and processes updates to the model parameters
using the gradients calculated by the worker nodes. The optimization algorithm used in
this project to find the neural network parameters (i.e.: the weights and biases) is
Stochastic Gradient Descent (SGD), which is the most commonly used algorithm for
performing parameter updates in neural networks. The traditional definition of SGD is
inherently serial, which is why I use a parallel version of SGD in this project. There are
two parameter update paradigms in parallel SGD: synchronous and asynchronous. In
case of synchronous parameter update the parameter server waits
to receive computed gradients from all worker nodes to update the global model
parameters, which are then broadcasted to all worker nodes to be utilized in the next
training iteration. On the other hand, asynchronous updates are processed by the
1parameter server immediately without waiting to receive gradients from all worker nodes
for the current training iteration. In this project I implement the synchronous version of
SGD. A serial implementation of the neural network is used as a benchmark to assess the
speed up from distributed training achieved via data parallelism with MPI and also
scalability.
