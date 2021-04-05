#include "NeuralNet.hpp" 
#include <string>
#include <fstream>

int main() {
    std::cout << "Hello World" << std::endl;
    
    // Number of classes
    int NUM_CLASSES = 3;

    // Create for neural network objects
    LayerDense dense_layer_1(2, 64);
    //std::cout << dense_layer_1.n_inputs << std::endl;
    ActivationRelu activation_relu();
    LayerDense dense_layer_2(64, NUM_CLASSES);
    // std::cout << dense_layer_2.n_inputs << std::endl;
    // std::cout << dense_layer_2.n_neurons << std::endl;
    ActivationSoftmax activation_softmax();
    Loss loss_categorical_crossentropy();
    StochasticGradientDescent optimizer_SGD(0.01, 5e-5);

    
}