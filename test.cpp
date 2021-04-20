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
    // std::cout << "dense1.biases is of size " << dense1.biases.rows() << "x" << dense1.biases.cols() << std::endl;
    // Eigen::MatrixXd weights(2, 3);
    // weights.row(0) << 1.0, -1.0, 0.5;
    // weights.row(1) << 0.5, -0.1, 2.0;
    // dense1.weights = weights;
    // dense1.forward(test_in.transpose());
    // test_out = dense1.output;
    // std::cout << "dense1.weights" << "\n" << dense1.weights << std::endl;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test relu backward - OK
    // ActivationRelu activation_relu;
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4.1, -5, 6;
    // Eigen::MatrixXd inputs(2, 3);
    // inputs.row(0) << 1, 2, 0;
    // inputs.row(1) << 0, 5, -6;
    // activation_relu.inputs = inputs;
    // activation_relu.backward(test_in);
    // test_out = activation_relu.dinputs;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test relu forward - OK
    // ActivationRelu activation_relu;
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out;
    // test_in.row(0) << 1, 2, -3;
    // test_in.row(1) << 4.1, -5, 6;
    // activation_relu.forward(test_in);
    // test_out = activation_relu.output;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test loss forward - OK
    // CrossEntropyLoss loss_categorical_crossentropy;
    // Eigen::MatrixXd y_pred(4, 3);
    // Eigen::VectorXd y_true(4);
    // Eigen::MatrixXd test_out;
    // y_pred.row(0) << 0.8, 0.1, 0.1;
    // y_pred.row(1) << 0.4, 0.6, 0.9;
    // y_pred.row(2) << 0.6, 0.1, 0.3;
    // y_pred.row(3) << 0.5, 0.4, 0.1;
    // std::cout << "y_pred" << "\n" << y_pred << std::endl;
    // y_true << 2, 2, 0, 1;
    // std::cout << "y_true" << "\n" << y_true << std::endl;
    // test_out = loss_categorical_crossentropy.forward(y_pred, y_true);
    // std::cout << "test_out" << "\n" << test_out << std::endl;
    // std::cout << "test_out" << "\n" << test_out.rows()<< "x" << test_out.cols() << std::endl;

    // test loss calculate - OK
    // CrossEntropyLoss loss_categorical_crossentropy;
    // Eigen::MatrixXd y_pred(4, 3);
    // Eigen::VectorXd y_true(4);
    // double test_out;
    // y_pred.row(0) << 0.8, 0.1, 0.1;
    // y_pred.row(1) << 0.4, 0.6, 0.9;
    // y_pred.row(2) << 0.6, 0.1, 0.3;
    // y_pred.row(3) << 0.5, 0.4, 0.1;
    // std::cout << "y_pred" << "\n" << y_pred << std::endl;
    // y_true << 2, 2, 0, 1;
    // std::cout << "y_true" << "\n" << y_true << std::endl;
    // test_out = loss_categorical_crossentropy.calculate(y_pred, y_true);
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test loss backward - OK
    // CrossEntropyLoss loss_categorical_crossentropy;
    // Eigen::MatrixXd dvalues(4, 3);
    // Eigen::VectorXd y_true(4);
    // Eigen::MatrixXd test_out;
    // dvalues.row(0) << 0.8, 0.1, 0.1;
    // dvalues.row(1) << 0.4, 0.6, 0.9;
    // dvalues.row(2) << 0.6, 0.1, 0.3;
    // dvalues.row(3) << 0.5, 0.4, 0.1;
    // std::cout << "dvalues" << dvalues.rows()<< "x" << dvalues.cols() << std::endl;
    // std::cout << "dvalues" << "\n" << dvalues << std::endl;
    // y_true << 2, 2, 0, 1;
    // std::cout << "y_true" << "\n" << y_true << std::endl;
    // loss_categorical_crossentropy.backward(dvalues, y_true);
    // test_out = loss_categorical_crossentropy.dinputs;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test optimizer pre_update_params - OK
    // StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);
    // double test_out;
    // optimizer_SGD.iterations = 500;
    // optimizer_SGD.pre_update_params(1.0);
    // test_out = optimizer_SGD.learning_rate;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test optimizer post_update_params - OK
    // StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);
    // double test_out;
    // optimizer_SGD.iterations = 500;
    // optimizer_SGD.post_update_params();
    // test_out = optimizer_SGD.iterations;
    // std::cout << "test_out" << "\n" << test_out << std::endl;

    // test optimizer update_params - OK
    // LayerDense dense1(2, 3);
    // Eigen::MatrixXd test_in(2, 3);
    // Eigen::MatrixXd test_out_w;
    // Eigen::VectorXd test_out_b;
    // test_in.row(0) << 1, 2, 3;
    // test_in.row(1) << 4, 5, 6;

    // Eigen::MatrixXd weights(2, 3);
    // weights.row(0) << 1.0, -1.0, 0.5;
    // weights.row(1) << 0.5, -0.1, 2.0;
    // dense1.weights = weights;

    // Eigen::MatrixXd dweights(2, 3);
    // dweights.row(0) << 0.1, -0.1, 0.5;
    // dweights.row(1) << 0.7, -0.2, 1.1;
    // dense1.dweights = dweights;

    // Eigen::VectorXd biases(3);
    // biases << 0.02, -0.3, 0.56;
    // dense1.biases = biases;

    // Eigen::VectorXd dbiases(3);
    // dbiases << 1.3, -1.6, 0.44;
    // dense1.dbiases = dbiases;

    // StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);
    // Eigen::MatrixXd test_out;
    // optimizer_SGD.iterations = 500;
    // optimizer_SGD.update_params(dense1);
    // test_out_w = dense1.weights;
    // std::cout << "test_out_w" << "\n" << test_out_w << std::endl;
    // test_out_b = dense1.biases;
    // std::cout << "test_out_b" << "\n" << test_out_b << std::endl;
}   
