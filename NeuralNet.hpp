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

        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd weights = Eigen::MatrixXd::Random(n_inputs,n_neurons) * 0.01; // initialize weights
        Eigen::VectorXd biases = Eigen::VectorXd::Zero(n_neurons); // initialize biases
        Eigen::MatrixXd output;

        Eigen::MatrixXd dweights; // derivative wrt weights
        Eigen::VectorXd dbiases; // derivative wrt biases
        Eigen::MatrixXd dinputs; // derivative wrt inputs

        // constructor
        LayerDense(int n_inputs, int n_neurons) {
            this->n_inputs = n_inputs;
            this->n_neurons = n_neurons;
        } 

        // Member functions declaration
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
};

class ActivationRelu {
    public:
        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // Member functions declaration
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
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
        void backward(Eigen::MatrixXd dvalues, Eigen::MatrixXd y_true);
};

class StochasticGradientDescent {
    public:
        double learning_rate; // learning rate
        double decay; // decay
        int iterations = 0; // initialize number of iterations to 0

        // constructor
        StochasticGradientDescent(double learning_rate, double decay) {
            this->learning_rate = learning_rate;
            this->decay = decay;
        } 

        // Member functions declaration
        void pre_update_params();
        void update_params(LayerDense layer);
        void post_update_params();
};

#endif // NEURALNET_HPP