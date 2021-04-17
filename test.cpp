#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>

int main() {
    std::cout << "Hello World" << std::endl;

    // test softmax forward - OK
    // ActivationSoftmax activation_softmax;
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4, 5, 6;
    // // std::cout << "test_in" << test_in << std::endl;
    // activation_softmax.forward(test_in);
    // test_out = activation_softmax.output;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test softmax backward - OK
    // ActivationSoftmax activation_softmax;
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd output(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4, 5, 6;
    // std::cout << "test_in" << "\n" << test_in << std::endl;
    // output.row(0) << 1, 2, 3;
    // output.row(1) << 4, 5, 6;
    // activation_softmax.output = output;
    // std::cout << "activation_softmax.output" << "\n" << activation_softmax.output << std::endl;
    // activation_softmax.backward(test_in);
    // test_out = activation_softmax.dinputs;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test dense backward - OK
    // LayerDense dense1(2, 3);
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4, 5, 6;
    // Eigen::MatrixXd inputs(2, 3);
    // inputs.row(0) << 1, 2, 3;
    // inputs.row(1) << 4, 5, 6;
    // dense1.inputs = inputs;
    // dense1.backward(test_in);
    // test_out = dense1.dweights;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test dense forward - OK
    // LayerDense dense1(2, 3);
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4, 5, 6;
    // Eigen::MatrixXd weights(2, 3);
    // weights.row(0) << 1.0, -1.0, 0.5;
    // weights.row(1) << 0.5, -0.1, 2.0;
    // dense1.weights = weights;
    // dense1.forward(test_in.transpose());
    // test_out = dense1.output;
    // std::cout << "dense1.weights" << "\n" << dense1.weights << std::endl;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test relu backward - OK
    ActivationRelu activation_relu;
    Eigen::MatrixXd test_in(2, 3);
    Eigen::MatrixXd test_out;
    test_in.row(0) << 1, 2, 3;
    test_in.row(1) << 4.1, -5, 6;
    Eigen::MatrixXd inputs(2, 3);
    inputs.row(0) << 1, 2, 0;
    inputs.row(1) << 0, 5, -6;
    activation_relu.inputs = inputs;
    activation_relu.backward(test_in);
    test_out = activation_relu.dinputs;
    std::cout << "test_out" << "\n" << test_out << std::endl;
}    