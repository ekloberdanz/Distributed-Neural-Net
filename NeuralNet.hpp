#ifndef NEURALNET_HPP
#define NEURALNET_HPP


#include <eigen3/Eigen/Eigen> 
#include <iostream> 
#include <vector>
#include <string>
#include <boost/range/combine.hpp>
  
// # Dense layer
// class Layer_Dense:
//     # Layer initialization
//     def __init__(self, n_inputs, n_neurons):
//         # Initialize weights and biases
//         np.random.seed(42)
//         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
//         print(self.weights)
//         self.biases = np.zeros((1, n_neurons))
//     # Forward pass
//     def forward(self, inputs):
//         # Remember input values
//         self.inputs = inputs
//         # Calculate output values from input ones, weights and biases
//         self.output = np.dot(inputs, self.weights) + self.biases
//     # Backward pass
//     def backward(self, dvalues):
//         # Gradients on parameters
//         self.dweights = np.dot(self.inputs.T, dvalues)
//         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
//         # Gradient on values
//         self.dinputs = np.dot(dvalues, self.weights.T)

class LayerDense {
    public:
        int n_inputs; // number of inputs
        int n_neurons; // number of neurons

        Eigen::VectorXd inputs; // inputs
        Eigen::MatrixXd weights = Eigen::MatrixXd::Random(n_inputs,n_neurons) * 0.01; // initialize weights
        Eigen::VectorXd biases = Eigen::VectorXd::Zero(n_neurons); // initialize biases

        Eigen::MatrixXd dweights; // derivative wrt weights
        Eigen::VectorXd dbiases; // derivative wrt biases
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // constructor
        LayerDense(int n_inputs, int n_neurons) {
            this->n_inputs = n_inputs;
            this->n_neurons = n_neurons;
        } 

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};

// Member functions definitions
void LayerDense::forward(Eigen::VectorXd inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd output = inputs * weights + biases;
}

void LayerDense::backward(Eigen::VectorXd dvalues) {
    Eigen::VectorXd dweights = inputs.transpose() * dvalues;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    Eigen::VectorXd dinputs = weights.transpose() * dvalues;
}

// # ReLU activation
// class Activation_ReLU:
//     # Forward pass
//     def forward(self, inputs):
//         # Remember input values
//         self.inputs = inputs
//         # Calculate output values from inputs
//         self.output = np.maximum(0, inputs)
//     # Backward pass
//     def backward(self, dvalues):
//         # Since we need to modify original variable,
//         # let’s make a copy of values first
//         self.dinputs = dvalues.copy()
//         # Zero gradient where input values were negative
//         self.dinputs[self.inputs <= 0] = 0

class ActivationRelu {
    public:
        Eigen::VectorXd inputs; // inputs
        Eigen::VectorXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // Member functions declaration
        void forward(Eigen::VectorXd inputs);
        void backward(Eigen::VectorXd dvalues);
};

// Member functions definitions
void ActivationRelu::forward(Eigen::VectorXd in) {
    inputs = in;
    output = (inputs.array() < 0).select(0, inputs);
}

void ActivationRelu::backward(Eigen::VectorXd dvalues) {
    dinputs = (inputs.array() < 0).select(0, dvalues);
}


// # Softmax activation
// class Activation_Softmax:
//   # Forward pass
//   def forward(self, inputs):
//     # Remember input values
//     self.inputs = inputs
//     # Get unnormalized probabilities
//     exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
//     # Normalize them for each sample
//     probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
//     self.output = probabilities
//   # Backward pass
//   def backward(self, dvalues):
//     # Create uninitialized array
//     self.dinputs = np.empty_like(dvalues)
//     # Enumerate outputs and gradients
//     for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
//       # Flatten output array
//       single_output = single_output.reshape(-1, 1)
//       # Calculate Jacobian matrix of the output
//       jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
//       # Calculate sample-wise gradient and add it to the array of sample gradients
//       self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class ActivationSoftmax {
    public:
        Eigen::MatrixXd inputs; // inputs
        Eigen::MatrixXd dinputs; // derivative wrt inputs
        Eigen::MatrixXd output;

        // Member functions declaration
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
};

// Member functions definitions
void ActivationSoftmax::forward(Eigen::MatrixXd in) {
    inputs = in;
    Eigen::MatrixXd exp_values;
    Eigen::MatrixXd max_values;
    Eigen::MatrixXd probabilities;

    max_values.setZero();
    max_values.diagonal() = inputs.rowwise().maxCoeff(); // max
    exp_values = (inputs - max_values).array().exp(); // unnormalized probabilities
    probabilities = exp_values.array().rowwise()  / (exp_values.array().colwise().sum()); // normalized probabilities
    output = probabilities;
}

void ActivationSoftmax::backward(Eigen::MatrixXd dvalues) {
    Eigen::MatrixXd jacobian_matrix;
    size_t i;
    Eigen::MatrixXd single_output;
    Eigen::MatrixXd single_dvalues;

    //for (i, (single_output, single_dvalues) in enum boost::combine(output, dvalues); i++) {
    for (i = 0; i <= dvalues.size(); i++)    {
        single_output = output.row(i);
        single_dvalues = dvalues.row(i);
        jacobian_matrix.setZero();
        jacobian_matrix.diagonal() = single_output - (single_output, single_output.transpose()).colwise().sum(); //Calculate Jacobian matrix of the output
        inputs.row(i) = (jacobian_matrix, single_dvalues).colwise().sum(); // Calculate sample-wise gradient and add it to the array of sample gradients
        }    
}

// # Common loss class
// class Loss:
//     # Calculates the data and regularization losses
//     # given model output and ground truth values
//     def calculate(self, output, y):
//         # Calculate sample losses
//         sample_losses = self.forward(output, y)
//         # Calculate mean loss
//         data_loss = np.mean(sample_losses)
//         # Return loss
//         return data_loss
//         # Cross-entropy loss

    // # Forward pass
    // def forward(self, y_pred, y_true):
    //     # Number of samples in a batch
    //     samples = len(y_pred)
    //     # Clip data to prevent division by 0
    //     # Clip both sides to not drag mean towards any value
    //     y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    //     # Probabilities for target values - only if categorical labels
    //     if len(y_true.shape) == 1:
    //       correct_confidences = y_pred_clipped[range(samples),y_true]
    //     # Mask values - only for one-hot encoded labels
    //     elif len(y_true.shape) == 2: 
    //       correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
    //     # Losses
    //     negative_log_likelihoods = -np.log(correct_confidences)
    //     return negative_log_likelihoods

    //     # Backward pass
    // def backward(self, dvalues, y_true):
    //     # Number of samples
    //     samples = len(dvalues)
    //     # Number of labels in every sample We’ll use the first sample to count them
    //     labels = len(dvalues[0])
    //     # If labels are sparse, turn them into one-hot vector
    //     if len(y_true.shape) == 1:
    //       y_true = np.eye(labels)[y_true]
    //     # Calculate gradient
    //     self.dinputs = -y_true / dvalues
    //     # Normalize gradient
    //     self.dinputs = self.dinputs / samples


class Loss {
    public:
        Eigen::MatrixXd dinputs;

        // Member functions declaration
        Eigen::VectorXd calculate(Eigen::VectorXd output, Eigen::VectorXd y);
        Eigen::VectorXd forward(Eigen::VectorXd y_pred, Eigen::VectorXd y_true);
        void backward(Eigen::VectorXd dvalues, Eigen::VectorXd y_true);
};

// Loss& loss();

// Member functions definitions
Eigen::VectorXd Loss::forward(Eigen::VectorXd y_pred, Eigen::MatrixXd y_true) {
    int samples = y_pred.size();
    //if y_true.rows() 
    Eigen::MatrixXd correct_confidences = (y_pred, y_true).colwise().sum();
    Eigen::MatrixXd negative_log_likelihoods = correct_confidences.array().log();
    return negative_log_likelihoods;
}

void Loss::backward(Eigen::VectorXd dvalues, Eigen::VectorXd y_true) {
    int samples = dvalues.size();
    dinputs = - (y_true.array().rowwise()/dvalues);
    dinputs = dinputs/samples;

}

Eigen::VectorXd Loss::calculate(Eigen::VectorXd output, Eigen::VectorXd y) {
    Eigen::VectorXd sample_losses = Loss::forward(output, y);
    Eigen::VectorXd data_loss = sample_losses.rowwise().mean();
    return sample_losses;
}



#endif // NEURALNET_HPP