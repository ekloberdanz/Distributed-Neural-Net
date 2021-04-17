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

    // test softmax backward
    ActivationSoftmax activation_softmax;
    Eigen::MatrixXd test_in(2, 3);
    Eigen::MatrixXd output(2, 3);
    Eigen::MatrixXd test_out;
    test_in.row(0) << 1, 2, 3;
    test_in.row(1) << 4, 5, 6;
    std::cout << "test_in" << "\n" << test_in << std::endl;
    output.row(0) << 1, 2, 3;
    output.row(1) << 4, 5, 6;
    activation_softmax.output = output;
    std::cout << "activation_softmax.output" << "\n" << activation_softmax.output << std::endl;
    activation_softmax.backward(test_in);
    test_out = activation_softmax.dinputs;
    std::cout << "test_out" << "\n" << test_out << std::endl;
}    