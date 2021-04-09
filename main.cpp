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
    // std::string MNIST_DATA_LOCATION = "dataset";
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    //     mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // // normalized_dataset = normalize_dataset(dataset);

    // // std::vector<unsigned char X_train = dataset.training_images.data();
    // // Eigen::MatrixXf test = Eigen::Map<Eigen::Matrix<float, 3, 1> >(test_vector.data());    

    // // Eigen:VectorXd X_train(dataset.training_images.data());

    // std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    // std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    // std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    // std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    // std::cout <<  typeid(dataset.training_images).name() << std::endl;

    // matrix to be loaded from a file
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test;

    // variables
    double loss;
     Eigen::VectorXd predictions;
 
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
        // get predictions



    }

    
}