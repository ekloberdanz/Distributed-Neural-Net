#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>

int main() {
    std::cout << "Hello World" << std::endl;
    
    // Load dataset
    Eigen::MatrixXd X_train;
    Eigen::VectorXi y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXi y_test;
 
    // load the training and testing data from the file
    X_train = load_matrix_data("./data/X_train.csv");
    y_train = load_vector_data("./data/y_train.csv");
    X_test = load_matrix_data("./data/X_test.csv");
    y_test = load_vector_data("./data/y_test.csv");
     
    // print the matrix in console
    // std::cout << X_train;
    std::cout << "Type of y_train " <<   typeid(y_train).name() << std::endl;
    std::cout << "Type of X_train " <<   typeid(X_train).name() << std::endl;
    std::cout << "The matrix X_train is of size " << X_train.rows() << "x" << X_train.cols() << std::endl;
    std::cout << "The vector y_train is of size " << y_train.rows() << "x" << y_train.cols() << std::endl;

    // std::cout << "X_train " << X_train << std::endl;
    // std::cout << "y_train " << y_train << std::endl;

    // Parameters
    int NUM_CLASSES = 3;
    // double decay = 1e-3;
    // double decay = 0.0;
    // double momentum = 0.0;
    double start_learning_rate = 1.0;

    Eigen::MatrixXd w_1;
    Eigen::MatrixXd w_2;
    w_1 = load_matrix_data("./data/weights_1.csv");
    w_2 = load_matrix_data("./data/weights_2.csv");

    // Create for neural network objects
    LayerDense dense_layer_1(2, 64, w_1);
    ActivationRelu activation_relu;
    LayerDense dense_layer_2(64, NUM_CLASSES, w_2);
    ActivationSoftmax activation_softmax;
    CrossEntropyLoss loss_categorical_crossentropy;
    StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);

    std::cout << "The matrix dense_layer_1.weights is of size " << dense_layer_1.weights.rows() << "x" << dense_layer_1.weights.cols() << std::endl;
    std::cout << "The matrix dense_layer_1.biases is of size " << dense_layer_1.biases.rows() << "x" << dense_layer_1.biases.cols() << std::endl;
    std::cout << "The matrix dense_layer_2.weights is of size " << dense_layer_2.weights.rows() << "x" << dense_layer_2.weights.cols() << std::endl;
    std::cout << "The matrix dense_layer_2.biases is of size " << dense_layer_2.biases.rows() << "x" << dense_layer_2.biases.cols() << std::endl;

    // std::cout << dense_layer_1.weights.row(1) << std::endl;
    // std::cout << dense_layer_1.biases << std::endl;
    // std::cout << dense_layer_2.weights << std::endl;
    // std::cout << dense_layer_2.biases << std::endl;

    // variables
    double loss;
    double train_accuracy;
    double test_accuracy;

    // Train DNN
    int NUMBER_OF_EPOCHS = 1;
    for (int epoch : boost::irange(1,NUMBER_OF_EPOCHS+1)) {
        // std::cout << epoch << "\n";

        ////////////////////////////////////////////////////////forward pass//////////////////////////////////////////////////////////////////////////////////
        // std::cout << "Layer 1" << "\n";
        dense_layer_1.forward(X_train);
        std::cout << "\nLayer 1 inputs in forward \n" << dense_layer_1.inputs.row(1) << std::endl;
        // std::cout << "The matrix X_train is of size " << X_train.rows() << "x" << X_train.cols() << std::endl;
        // std::cout << "\nThe matrix dense_layer_1.output " << dense_layer_1.output.row(1) << std::endl;
        // std::cout << "\nThe matrix dense_layer_1.output is of size " << dense_layer_1.output.rows() << "x" << dense_layer_1.output.cols() << std::endl;
        activation_relu.forward(dense_layer_1.output);
        // std::cout << "The matrix activation_relu.output is of size " << activation_relu.output.rows() << "x" << activation_relu.output.cols() << std::endl;
        // std::cout << "The matrix activation_relu.output " << activation_relu.output << std::endl;
        // std::cout << "Layer 2" << "\n";
        dense_layer_2.forward(activation_relu.output);
        std::cout << "\nLayer 2 inputs in forward \n" << dense_layer_2.inputs.row(1) << std::endl;
        // std::cout << "\nThe matrix dense_layer_2.output " << dense_layer_2.output.row(1) << std::endl;
        // std::cout << "The matrix dense_layer_2.output is of size " << dense_layer_2.output.rows() << "x" << dense_layer_2.output.cols() << std::endl;
        // std::cout << "The matrix dense_layer_2.output " << dense_layer_2.output << std::endl;
        activation_softmax.forward(dense_layer_2.output);
        // std::cout << "The matrix activation_softmax.output is of size " << activation_softmax.output.rows() << "x" << activation_softmax.output.cols() << std::endl;
        // calculate loss
        loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_train);
        std::cout << "loss: " << loss << std::endl;
       // get predictions and accuracy
        Eigen::MatrixXd::Index maxRow, maxCol;
        Eigen::VectorXi predictions(activation_softmax.output.rows());
        Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());
        double pred;
        int index_pred;

        // Eigen::VectorXi predictions_int = predictions.cast <int> ();
        // Eigen::VectorXi y_train_int = y_train.cast <int> ();


        for (int i=0; i < activation_softmax.output.rows(); i++) {
            // std::cout << "i: " << i << std::endl;
            pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
            // std::cout << "pred: " << pred << std::endl;
            index_pred = maxCol;
            // std::cout << "index_pred: " << index_pred << std::endl;
            predictions(i) = index_pred;
            // std::cout << "predictions_int(i): " << predictions_int(i) << std::endl;
            // std::cout << "y_train_int(i): " << y_train_int(i) << std::endl;
            pred_truth_comparison(i) = predictions(i) == y_train(i);
            // std::cout << "pred_truth_comparison(i): " << pred_truth_comparison(i) << std::endl;
        }
        // std::cout << predictions_int << std::endl;
        // std::cout << dense_layer_1.output << std::endl;
        // std::cout << activation_relu.output << std::endl;
        // std::cout << dense_layer_2.output << std::endl;
        // std::cout << activation_softmax.output << std::endl;

        // std::cout << "The vector predictions is of size " << predictions_int.rows() << "x" << predictions_int.cols() << std::endl;
        // std::cout << "The vector pred_truth_comparison is of size " << pred_truth_comparison.rows() << "x" << pred_truth_comparison.cols() << std::endl;
        train_accuracy = pred_truth_comparison.mean();
        // std::cout << "train_accuracy: " << train_accuracy << std::endl;
        // std::cout << "loss: " << loss << std::endl;

        if (epoch % 100 == 0) {
            std::cout << "epoch: " << epoch << std::endl;
            std::cout << "train_accuracy: " << train_accuracy << std::endl;
            std::cout << "learning_rate: " << optimizer_SGD.learning_rate << std::endl;
            std::cout << "loss: " << loss << std::endl;
            // std::cout << "dense_layer_1.weights: " << dense_layer_1.weights.mean() << std::endl;
            // std::cout << "dense_layer_1.output: " << dense_layer_1.output.mean() << std::endl;
            // std::cout << "dense_layer_2.output: " << dense_layer_2.output.mean() << std::endl;
            // std::cout << "activation_relu.output: " << activation_relu.output << std::endl;
            // std::cout << "activation_softmax.output: " << activation_softmax.output.mean() << std::endl;
            // std::cout << "dense_layer_2.dweights: " << dense_layer_2.weights.mean() << std::endl;
            // std::cout << "dense_layer_2.biases: " << dense_layer_2.biases.mean() << std::endl;
            // std::cout << activation_softmax.output.mean() << std::endl;
            // std::cout << "loss_categorical_crossentropy.dinputs" << loss_categorical_crossentropy.dinputs.mean() << std::endl;
            // std::cout << "dense_layer_2.dinputs: " << dense_layer_2.dinputs.mean() << std::endl;
            // std::cout << "activation_softmax.dinputs: " << activation_softmax.dinputs.mean() << std::endl;
            // std::cout << "activation_softmax.output" << activation_softmax.output << std::endl;
        }
        // std::cout << "activation_softmax.output " << activation_softmax.output.row(1) << std::endl;
        // std::cout << "true label " << y_train.row(1) << std::endl;
        ////////////////////////////////////////////////////////////backward pass/////////////////////////////////////////////////////////////////////////
        loss_categorical_crossentropy.backward(activation_softmax.output, y_train);
        // std::cout << "The matrix loss_categorical_crossentropy.dinputs is of size " << loss_categorical_crossentropy.dinputs.rows() << "x" << loss_categorical_crossentropy.dinputs.cols() << std::endl;
        // std::cout << "loss_categorical_crossentropy.dinputs" << loss_categorical_crossentropy.dinputs.mean() << std::endl;
        // std::cout << "\ndinputs for softmax \n" << loss_categorical_crossentropy.dinputs << std::endl;
        activation_softmax.backward(loss_categorical_crossentropy.dinputs);
        // std::cout << "The matrix activation_softmax.dinputs is of size " << activation_softmax.dinputs.rows() << "x" << activation_softmax.dinputs.cols() << std::endl;
        // std::cout << "\nLayer 2 inputs \n" << dense_layer_2.inputs.row(1) << std::endl;
        std::cout << "\ndinputs for dense 2 \n" << activation_softmax.dinputs.row(0) << std::endl;
        dense_layer_2.backward(activation_softmax.dinputs);
        // std::cout << "The matrix dense_layer_2.dinputs is of size " << dense_layer_2.dinputs.rows() << "x" << dense_layer_2.dinputs.cols() << std::endl;
        activation_relu.backward(dense_layer_2.dinputs);
        // std::cout << "The matrix activation_relu.dinputs is of size " << activation_relu.dinputs.rows() << "x" << activation_relu.dinputs.cols() << std::endl;
        // std::cout << "\nLayer 1 inputs \n" << dense_layer_1.inputs.row(1) << std::endl;
        dense_layer_1.backward(activation_relu.dinputs);
        // std::cout << "The matrix dense_layer_1.dinputs is of size " << dense_layer_1.dinputs.rows() << "x" << dense_layer_1.dinputs.cols() << std::endl;

        // std::cout << "dense_layer_1.biases: " << dense_layer_2.biases << std::endl;
        // std::cout << "dense_layer_1.dbiases: " << dense_layer_2.biases << std::endl;
        // std::cout << "activation_softmax.dinputs: " << activation_softmax.dinputs.row(1) << std::endl;

        ////////////////////////////////////////////////////////////optimizer - update weights and biases/////////////////////////////////////////////////////////////////////////
        optimizer_SGD.pre_update_params(start_learning_rate);
        // std::cout << "learning_rate: " << optimizer_SGD.learning_rate << std::endl;
        // std::cout << "\nLayer 2 dweights \n" << dense_layer_2.dweights << std::endl;
        // std::cout << "\nLayer 2 dbiases \n" << dense_layer_2.dbiases << std::endl;
        // std::cout << "\nLayer 2 weights before \n" << dense_layer_2.weights << std::endl;
        optimizer_SGD.update_params(dense_layer_1);
        // std::cout << "\nLayer 2 weights after \n" << dense_layer_2.weights << std::endl;
        // std::cout << "Layer 1 biases" << dense_layer_1.biases.mean() << std::endl;

        optimizer_SGD.update_params(dense_layer_2);
        // std::cout << "Layer 2 weights" << dense_layer_2.weights.mean() << std::endl;
        // std::cout << "Layer 2 biases"<< dense_layer_2.biases.mean() << std::endl;

        optimizer_SGD.post_update_params();
        // std::cout << "iterations: " << optimizer_SGD.iterations << std::endl;

    }
    // // Test DNN
    // dense_layer_1.forward(X_test);
    // // std::cout << "The matrix dense_layer_1.output is of size " << dense_layer_1.output.rows() << "x" << dense_layer_1.output.cols() << std::endl;
    // activation_relu.forward(dense_layer_1.output);
    // // std::cout << "The matrix activation_relu.output is of size " << activation_relu.output.rows() << "x" << activation_relu.output.cols() << std::endl;
    // dense_layer_2.forward(activation_relu.output);
    // // std::cout << "The matrix dense_layer_2.output is of size " << dense_layer_2.output.rows() << "x" << dense_layer_2.output.cols() << std::endl;
    // activation_softmax.forward(dense_layer_2.output);
    // // std::cout << "The matrix activation_softmax.output is of size " << activation_softmax.output.rows() << "x" << activation_softmax.output.cols() << std::endl;
    // // calculate loss
    // loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_test);
    // // get predictions and accuracy
    // Eigen::MatrixXd::Index maxRow, maxCol;
    // Eigen::VectorXi predictions(activation_softmax.output.rows());
    // Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());
    // double pred;
    // int index_pred;
    // // Eigen::VectorXi predictions_int = predictions.cast <int> ();
    // // Eigen::VectorXi y_test_int = y_test.cast <int> ();

    // for (int i=0; i < activation_softmax.output.rows(); i++) {
    //         // std::cout << "i: " << i << std::endl;
    //         pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
    //         index_pred = maxCol;
    //         predictions(i) = index_pred;
    //         pred_truth_comparison(i) = predictions(i) == y_test(i);
    // }
    // // std::cout << "The vector predictions is of size " << predictions.rows() << "x" << predictions.cols() << std::endl;
    // // std::cout << "The vector pred_truth_comparison is of size " << pred_truth_comparison.rows() << "x" << pred_truth_comparison.cols() << std::endl;
    // test_accuracy = pred_truth_comparison.mean();
    // std::cout << "test_accuracy: " << test_accuracy << std::endl;
}