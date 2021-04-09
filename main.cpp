#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
// #include "mnist/mnist_reader.hpp"
// #include "mnist/mnist_utils.hpp"
// #include "mnist/mnist_reader_common.hpp"
// #include "mnist/mnist_reader_less.hpp"
#include <boost/range/irange.hpp>
#include <typeinfo>

int main() {
    std::cout << "Hello World" << std::endl;
    
    // Load dataset
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test;
 
    // load the training and testing data from the file
    X_train = load_matrix_data("./data/X_train.csv");
    y_train = load_vector_data("./data/y_train.csv");
    X_test = load_matrix_data("./data/X_test.csv");
    y_test = load_vector_data("./data/y_test.csv");
     
    // print the matrix in console
    // std::cout << X_train;
    std::cout << "The matrix X_train is of size " << X_train.rows() << "x" << X_train.cols() << std::endl;

    // Number of classes
    int NUM_CLASSES = 3;

    // Create for neural network objects
    LayerDense dense_layer_1(2, 64);
    ActivationRelu activation_relu;
    LayerDense dense_layer_2(64, NUM_CLASSES);
    ActivationSoftmax activation_softmax;
    Loss loss_categorical_crossentropy;
    StochasticGradientDescent optimizer_SGD(0.01, 5e-5);

    std::cout << "The matrix dense_layer_1.weights is of size " << dense_layer_1.weights.rows() << "x" << dense_layer_1.weights.cols() << std::endl;
    std::cout << "The matrix dense_layer_1.biases is of size " << dense_layer_1.biases.rows() << "x" << dense_layer_1.biases.cols() << std::endl;
    std::cout << "The matrix dense_layer_2.weights is of size " << dense_layer_2.weights.rows() << "x" << dense_layer_2.weights.cols() << std::endl;
    std::cout << "The matrix dense_layer_2.biases is of size " << dense_layer_2.biases.rows() << "x" << dense_layer_2.biases.cols() << std::endl;
    // std::cout << dense_layer_1.biases;


    // variables
    double loss;
    double train_accuracy;

    // Train DNN
    int NUMBER_OF_EPOCHS = 10;
    for (int epoch : boost::irange(1,NUMBER_OF_EPOCHS+1)) {
        std::cout << epoch << "\n";
        // forward pass
        dense_layer_1.forward(X_train);
        std::cout << "The matrix dense_layer_1.output is of size " << dense_layer_1.output.rows() << "x" << dense_layer_1.output.cols() << std::endl;
        activation_relu.forward(dense_layer_1.output);
        std::cout << "The matrix activation_relu.output is of size " << activation_relu.output.rows() << "x" << activation_relu.output.cols() << std::endl;
        dense_layer_2.forward(activation_relu.output);
        std::cout << "The matrix dense_layer_2.output is of size " << dense_layer_2.output.rows() << "x" << dense_layer_2.output.cols() << std::endl;
        activation_softmax.forward(dense_layer_2.output);
        std::cout << "The matrix activation_softmax.output is of size " << activation_softmax.output.rows() << "x" << activation_softmax.output.cols() << std::endl;
        // calculate loss
        loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_train);
        // get predictions and accuracy
        Eigen::MatrixXd::Index maxRow, maxCol;
        Eigen::VectorXd predictions(activation_softmax.output.cols());
        Eigen::VectorXd pred_truth_comparison(activation_softmax.output.cols());
        double pred;
        int index_pred;
        for (int i=0; i < activation_softmax.output.cols(); i++) {
            std::cout << "i: " << i << std::endl;
            pred = activation_softmax.output.col(i).maxCoeff(&maxRow, &maxCol);
            std::cout << "pred: " << pred << std::endl;
            index_pred = maxRow;
            std::cout << "index_pred: " << index_pred << std::endl;
            predictions(i) = index_pred;
            pred_truth_comparison(i) = predictions(i) == y_train(i);
        }
        std::cout << "The vector predictions is of size " << predictions.rows() << "x" << predictions.cols() << std::endl;
        std::cout << "The vector pred_truth_comparison is of size " << pred_truth_comparison.rows() << "x" << pred_truth_comparison.cols() << std::endl;
        train_accuracy = pred_truth_comparison.mean();
        std::cout << "train_accuracy: " << train_accuracy << std::endl;
    }

    
}