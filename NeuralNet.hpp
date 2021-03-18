#ifndef NEURALNET_HPP
#define NEURALNET_HPP


#include <eigen3/Eigen/Eigen> 
#include <iostream> 
#include <vector>
#include <string>
#include <boost/range/combine.hpp>
  

class LayerDense {
    public:
        int n_inputs; // number of inputs
        int n_neurons; // number of neurons

        Eigen::VectorXd inputs; // inputs
        Eigen::MatrixXd weights = Eigen::MatrixXd::Random(n_inputs,n_neurons) * 0.01; // initialize weights
        Eigen::VectorXd biases = Eigen::VectorXd::Zero(n_neurons); // initialize biases

        Eigen::MatrixXd dweights; // derivative wrt weights
        Eigen::VectorXd dbiases; // derivative wrt biases
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // constructor
        LayerDense(int n_inputs, int n_neurons) {
            this->n_inputs = n_inputs;
            this->n_neurons = n_neurons;
        } 

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};

class ActivationRelu {
    public:
        Eigen::VectorXd inputs; // inputs
        Eigen::VectorXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};


class ActivationSoftmax {
    public:
        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // Member functions declaration
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
};

class Loss {
    public:
        Eigen::MatrixXd dinputs;

        // Member functions declaration
        Eigen::MatrixXd calculate(Eigen::MatrixXd output, Eigen::MatrixXd y);
        Eigen::MatrixXd forward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true);
        void backward(Eigen::VectorXd dvalues, Eigen::VectorXd y_true);
};

// # SGD optimizer
// class Optimizer_SGD:
//     # Initialize optimizer - set settings,learning rate of 1. is default for this optimizer
//     def __init__(self, learning_rate=1., decay=0.):
//         self.learning_rate = learning_rate
//         self.current_learning_rate = learning_rate
//         self.decay = decay
//         self.iterations = 0
//     # Call once before any parameter updates
//     def pre_update_params(self):
//         if self.decay:
//             self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
//     # Update parameters
//     def update_params(self, layer):
//         layer.weights += -self.current_learning_rate * layer.dweights
//         layer.biases += -self.current_learning_rate * layer.dbiases
//     # Call once after any parameter updates
//     def post_update_params(self):
//         self.iterations += 1


#endif // NEURALNET_HPP