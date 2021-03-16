#ifndef NEURALNET_HPP
#define NEURALNET_HPP


#include <eigen3/Eigen/Eigen> 
#include <iostream> 
#include <vector>
#include <string>
  
// # Dense layer
// class Layer_Dense:
//     # Layer initialization
//     def __init__(self, n_inputs, n_neurons):
//         # Initialize weights and biases
//         np.random.seed(42)
//         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
//         print(self.weights)
//         self.biases = np.zeros((1, n_neurons))
//     # Forward pass
//     def forward(self, inputs):
//         # Remember input values
//         self.inputs = inputs
//         # Calculate output values from input ones, weights and biases
//         self.output = np.dot(inputs, self.weights) + self.biases
//     # Backward pass
//     def backward(self, dvalues):
//         # Gradients on parameters
//         self.dweights = np.dot(self.inputs.T, dvalues)
//         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
//         # Gradient on values
//         self.dinputs = np.dot(dvalues, self.weights.T)

class LayerDense {
    public:
        int n_inputs; // number of inputs
        int n_neurons; // number of neurons

        Eigen::VectorXd inputs; // inputs
        Eigen::MatrixXd weights = Eigen::MatrixXd::Random(n_inputs,n_neurons) * 0.01; // initialize weights
        Eigen::VectorXd biases = Eigen::VectorXd::Zero(n_neurons); // initialize biases

        Eigen::MatrixXd dvalues; // derivative values
        Eigen::MatrixXd dweights; // derivative wrt weights
        Eigen::VectorXd dbiases; // derivative wrt biases
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // constructor
        LayerDense(int n_inputs, int n_neurons) {
            n_inputs = n_inputs;
            n_neurons = n_neurons;
        } 

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};

// Member functions definitions
void LayerDense::forward(Eigen::VectorXd inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd output = inputs * weights + biases;
}

void LayerDense::backward(Eigen::VectorXd dvalues) {
    Eigen::VectorXd dweights = inputs.transpose() * dvalues;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    Eigen::VectorXd dinputs = weights.transpose() * dvalues;
}

// # ReLU activation
// class Activation_ReLU:
//     # Forward pass
//     def forward(self, inputs):
//         # Remember input values
//         self.inputs = inputs
//         # Calculate output values from inputs
//         self.output = np.maximum(0, inputs)
//     # Backward pass
//     def backward(self, dvalues):
//         # Since we need to modify original variable,
//         # let’s make a copy of values first
//         self.dinputs = dvalues.copy()
//         # Zero gradient where input values were negative
//         self.dinputs[self.inputs <= 0] = 0

class ActivationRelu {
    public:
        Eigen::VectorXd inputs; // inputs
        Eigen::VectorXd dvalues; // derivative values
        Eigen::VectorXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // constructor
        ActivationRelu() {
        } 

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};

// Member functions definitions
void ActivationRelu::forward(Eigen::VectorXd inputs) {
    this->inputs = inputs;
    Eigen::VectorXd output = inputs;
}

void ActivationRelu::backward(Eigen::VectorXd dvalues) {
    Eigen::VectorXd dweights = inputs.transpose() * dvalues;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    Eigen::VectorXd dinputs = weights.transpose() * dvalues;
}


#endif // NEURALNET_HPP