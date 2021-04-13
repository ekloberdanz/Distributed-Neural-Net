#ifndef NEURALNET_HPP
#define NEURALNET_HPP


#include <eigen3/Eigen/Eigen> 
#include <iostream> 
#include <vector>
#include <string>
#include <boost/range/combine.hpp>
#include <fstream>

class LayerDense {
    public:
        int n_inputs; // number of inputs
        int n_neurons; // number of neurons

        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd weights;
        Eigen::MatrixXd weight_momentums;
        Eigen::VectorXd biases;
        Eigen::VectorXd bias_momentums;
        Eigen::MatrixXd output;

        Eigen::MatrixXd dweights; // derivative wrt weights
        Eigen::VectorXd dbiases; // derivative wrt biases
        Eigen::MatrixXd dinputs; // derivative wrt inputs

        // constructor
        LayerDense(int n_inputs, int n_neurons) {
            this->weights = Eigen::MatrixXd::Random(n_inputs,n_neurons) * 0.01; // initialize weights
            this->biases = Eigen::VectorXd::Zero(n_neurons); // initialize biases
            this->n_inputs = n_inputs;
            this->n_neurons = n_neurons;
            this->weight_momentums = Eigen::MatrixXd::Zero(n_inputs,n_neurons); // weight momentum
            this->bias_momentums = Eigen::VectorXd::Zero(n_neurons); // bias momentum
            this->dweights = dweights;
            this->dbiases = dbiases;
            this->dinputs = dinputs;
            this->output = output;
            this->inputs = inputs;
        } 

        // Member functions declaration
        void forward(const Eigen::MatrixXd &inputs);
        void backward(const Eigen::MatrixXd &dvalues, const Eigen::MatrixXd &inputs);
};

class ActivationRelu {
    public:
        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;
        // dinputs = Eigen::MatrixXd:: Zero(dvalues.rows(),dvalues.cols());


        // Member functions declaration
        void forward(const Eigen::MatrixXd &inputs);
        void backward(const Eigen::MatrixXd &dvalues);
};


class ActivationSoftmax {
    public:
        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

    ActivationSoftmax(){
        this->dinputs = dinputs;
        this->output = output;

    }
        // Member functions declaration
        void forward(const Eigen::MatrixXd &inputs);
        void backward(const Eigen::MatrixXd &dvalues);
};

class CrossEntropyLoss {
    public:
        Eigen::MatrixXd dinputs;

        // Member functions declaration
        Eigen::VectorXd forward(const Eigen::MatrixXd &y_pred, const Eigen::VectorXd &y_true);
        double calculate(const Eigen::MatrixXd &output, const Eigen::VectorXd &y);
        void backward(const Eigen::MatrixXd &dvalues, const Eigen::VectorXd &y_true);
};

class StochasticGradientDescent {
    public:
        double learning_rate; // learning rate
        double decay; // decay
        double momentum; // momentum
        double iterations = 0.0; // initialize number of iterations to 0

        // constructor
        StochasticGradientDescent(double learning_rate, double decay, double momentum) {
            this->learning_rate = learning_rate;
            this->decay = decay;
            this->momentum = momentum;
            this->iterations = iterations;
        } 

        // Member functions declaration
        void pre_update_params(double start_learning_rate);
        void update_params(LayerDense &layer);
        void post_update_params();
};

Eigen::MatrixXd load_matrix_data(std::string fileToOpen);
Eigen::VectorXd load_vector_data(std::string fileToOpen);

#endif // NEURALNET_HPP