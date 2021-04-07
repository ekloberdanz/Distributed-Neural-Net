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
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd X_test;
    Eigen::MatrixXd y_test;
 
    // load the training and testing data from the file
    X_train = load_matrix_data("./data/X_train.csv");
    y_train = load_vector_data("./data/y_train.csv");
    X_test = load_matrix_data("./data/X_test.csv");
    y_test = load_vector_data("./data/y_test.csv");
     
    // print the matrix in console
    std::cout << y_test;

    // Number of classes
    int NUM_CLASSES = 3;

    // Create for neural network objects
    LayerDense dense_layer_1(2, 64);
    ActivationRelu activation_relu();
    LayerDense dense_layer_2(64, NUM_CLASSES);
    ActivationSoftmax activation_softmax();
    Loss loss_categorical_crossentropy();
    StochasticGradientDescent optimizer_SGD(0.01, 5e-5);

    // Train DNN
    int NUMBER_OF_EPOCHS = 10;
    for (int epoch : boost::irange(1,NUMBER_OF_EPOCHS+1)) {
        std::cout << epoch << "\n";
        // forward pass
        dense_layer_1.forward(X_train);
        activation_relu.forward(dense_layer_1.output)
        dense_layer_2.forward(activation_relu.output)
        activation_softmax.forward(dense_layer_2.output)
    }

    
}