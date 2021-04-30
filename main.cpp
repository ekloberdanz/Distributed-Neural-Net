#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>
#include <chrono>
#include <ctime>

int main() {
    // Dataset
    Eigen::MatrixXd X_train;
    Eigen::VectorXi y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXi y_test;
 
    // Load the training and testing data from the file
    //X_train = load_matrix_data("./data/X_train.csv");
    //y_train = load_vector_data("./data/y_train.csv");
    //X_test = load_matrix_data("./data/X_test.csv");
    //y_test = load_vector_data("./data/y_test.csv");

    X_train = load_matrix_data("./data/X_train_large.csv");
    y_train = load_vector_data("./data/y_train_large.csv");
    X_test = load_matrix_data("./data/X_test_large.csv");
    y_test = load_vector_data("./data/y_test_large.csv");
    
    std::cout << "The matrix X_train is of size " << X_train.rows() << "x" << X_train.cols() << std::endl;
    std::cout << "The vector y_train is of size " << y_train.rows() << "x" << y_train.cols() << std::endl;
    
    // Parameters
    int NUM_CLASSES = 4;
    double start_learning_rate = 1.0;

    // Load initial weights from csv file
    //Eigen::MatrixXd w_1;
    //Eigen::MatrixXd w_2;
    //w_1 = load_matrix_data("./data/weights_1.csv");
    //w_2 = load_matrix_data("./data/weights_2.csv");

    // Create for neural network objects
    LayerDense dense_layer_1(2, 64);
    ActivationRelu activation_relu;
    LayerDense dense_layer_2(64, NUM_CLASSES);
    ActivationSoftmax activation_softmax;
    CrossEntropyLoss loss_categorical_crossentropy;
    StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);

    // variables
    double loss;
    double train_accuracy;
    double test_accuracy;
    double pred;
    int index_pred;

    // Train DNN
    auto t_start = std::chrono::high_resolution_clock::now();
    
    int NUMBER_OF_EPOCHS = 10000;
    for (int epoch : boost::irange(0,NUMBER_OF_EPOCHS)) {
        ////////////////////////////////////////////////////////forward pass//////////////////////////////////////////////////////////////////////////////////
        dense_layer_1.forward(X_train);
        activation_relu.forward(dense_layer_1.output);
        dense_layer_2.forward(activation_relu.output);
        activation_softmax.forward(dense_layer_2.output);
        // calculate loss
        loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_train);
        // get predictions and accuracy
        Eigen::MatrixXd::Index maxRow, maxCol;
        Eigen::VectorXi predictions(activation_softmax.output.rows());
        Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

        for (int i=0; i < activation_softmax.output.rows(); i++) {
            pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
            index_pred = maxCol;
            predictions(i) = index_pred;
            pred_truth_comparison(i) = predictions(i) == y_train(i);
        }
        train_accuracy = pred_truth_comparison.mean();
        if (epoch % 1000 == 0) {
            std::cout << "epoch: " << epoch << std::endl;
            std::cout << "train_accuracy: " << train_accuracy << std::endl;
            std::cout << "learning_rate: " << optimizer_SGD.learning_rate << std::endl;
            std::cout << "loss: " << loss << std::endl;
        }

        ////////////////////////////////////////////////////////////backward pass/////////////////////////////////////////////////////////////////////////
        loss_categorical_crossentropy.backward(activation_softmax.output, y_train);
        activation_softmax.backward(loss_categorical_crossentropy.dinputs);
        dense_layer_2.backward(activation_softmax.dinputs);
        activation_relu.backward(dense_layer_2.dinputs);
        dense_layer_1.backward(activation_relu.dinputs);

        ////////////////////////////////////////////////////////////debugging/////////////////////////////////////////////////////////////////////////
        // std::cout << "Type of y_train " <<   typeid(y_train).name() << std::endl;
        // std::cout << "Type of X_train " <<   typeid(X_train).name() << std::endl;
        // std::cout << "The matrix X_train is of size " << X_train.rows() << "x" << X_train.cols() << std::endl;
        // std::cout << "The vector y_train is of size " << y_train.rows() << "x" << y_train.cols() << std::endl;
        // std::cout << "X_train " << X_train << std::endl;
        // std::cout << "y_train " << y_train << std::endl;
        
        // std::cout << "The matrix dense_layer_1.weights is of size " << dense_layer_1.weights.rows() << "x" << dense_layer_1.weights.cols() << std::endl;
        // std::cout << "The matrix dense_layer_1.biases is of size " << dense_layer_1.biases.rows() << "x" << dense_layer_1.biases.cols() << std::endl;
        // std::cout << "The matrix dense_layer_2.weights is of size " << dense_layer_2.weights.rows() << "x" << dense_layer_2.weights.cols() << std::endl;
        // std::cout << "The matrix dense_layer_2.biases is of size " << dense_layer_2.biases.rows() << "x" << dense_layer_2.biases.cols() << std::endl;

        // std::cout << dense_layer_1.weights << std::endl;
        // std::cout << dense_layer_1.biases << std::endl;
        // std::cout << dense_layer_2.weights << std::endl;
        // std::cout << dense_layer_2.biases << std::endl;
 
        ////////////////////////////////////////////////////////////optimizer - update weights and biases/////////////////////////////////////////////////////////////////////////
        optimizer_SGD.pre_update_params(start_learning_rate);
        optimizer_SGD.update_params(dense_layer_1);
        optimizer_SGD.update_params(dense_layer_2);
        optimizer_SGD.post_update_params();
        
        ////////////////////////////////////////////////////////////debugging/////////////////////////////////////////////////////////////////////////
        // std::cout << "\nepoch: " << epoch << "\n";

        // std::cout << "\nLayer 1 weights after \n" << dense_layer_1.weights << std::endl;
        // std::cout << "\nLayer 1 biases after \n" << dense_layer_1.biases << std::endl;
        // std::cout << "\nLayer 2 weights after \n" << dense_layer_2.weights << std::endl;
        // std::cout << "\nLayer 2 biases after \n" << dense_layer_2.biases << std::endl;

        // std::cout << "\ndense_layer_1.output\n" << dense_layer_1.output << std::endl;
        // std::cout << "\nactivation_relu.output\n" << activation_relu.output << std::endl;
        // std::cout << "\ndense_layer_2.output\n" << dense_layer_2.output << std::endl;
        // std::cout << "\nactivation_softmax.output\n" << activation_softmax.output << std::endl;

        // std::cout << "loss: " << loss << std::endl;
        // std::cout << "predictions: " << predictions << std::endl;

        // std::cout << "loss_categorical_crossentropy.dinputs: " << loss_categorical_crossentropy.dinputs << std::endl;
        // std::cout << "activation_softmax.dinputs: " << activation_softmax.dinputs << std::endl;
        // std::cout << "dense_layer_2.dinputs: " << dense_layer_2.dinputs << std::endl;
        // std::cout << "activation_relu.dinputs: " << activation_relu.dinputs << std::endl;
    }
    
    // Time training time
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "\ntraining took " << std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count()<< " seconds\n";

    // Test DNN
    dense_layer_1.forward(X_test);
    activation_relu.forward(dense_layer_1.output);
    dense_layer_2.forward(activation_relu.output);
    activation_softmax.forward(dense_layer_2.output);
    // calculate loss
    loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_test);
    // get predictions and accuracy
    Eigen::MatrixXd::Index maxRow, maxCol;
    Eigen::VectorXi predictions(activation_softmax.output.rows());
    Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

    for (int i=0; i < activation_softmax.output.rows(); i++) {
            pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
            index_pred = maxCol;
            predictions(i) = index_pred;
            pred_truth_comparison(i) = predictions(i) == y_test(i);
    }
    test_accuracy = pred_truth_comparison.mean();
    std::cout << "\ntest_accuracy: " << test_accuracy << std::endl;
}
