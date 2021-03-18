#include "NeuralNet.hpp"


// LayerDense functions definitions
void LayerDense::forward(Eigen::VectorXd inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd output = inputs * weights + biases;
}

void LayerDense::backward(Eigen::VectorXd dvalues) {
    Eigen::VectorXd dweights = inputs.transpose() * dvalues;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    Eigen::VectorXd dinputs = weights.transpose() * dvalues;
}

// ActivationRelu functions definitions
void ActivationRelu::forward(Eigen::VectorXd in) {
    inputs = in;
    output = (inputs.array() < 0).select(0, inputs);
}

void ActivationRelu::backward(Eigen::VectorXd dvalues) {
    dinputs = (inputs.array() < 0).select(0, dvalues);
}

// ActivationSoftmax functions definitions
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

// Loss functions definitions
Eigen::MatrixXd Loss::forward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true) {
    int samples = y_pred.size();
    //if y_true.rows() 
    Eigen::MatrixXd correct_confidences = (y_pred, y_true).colwise().sum();
    Eigen::MatrixXd negative_log_likelihoods = correct_confidences.array().log();
    return negative_log_likelihoods;
}

void Loss::backward(Eigen::VectorXd dvalues, Eigen::VectorXd y_true) {
    int samples = dvalues.size();
    Eigen::VectorXd ones;
    ones.fill(-1);
    dinputs = y_true.array() * (ones.array()/dvalues.array());
    dinputs = dinputs/samples;

}

Eigen::MatrixXd Loss::calculate(Eigen::MatrixXd output, Eigen::MatrixXd y) {
    Eigen::MatrixXd sample_losses = Loss::forward(output, y);
    Eigen::MatrixXd data_loss = sample_losses.rowwise().mean();
    return sample_losses;
}


// SGD functions declaration
void SGD::pre_update_params() {
}

void SGD::update_params(LayerDense layer) {
    layer.weights += learning_rate * layer.dweights;
    layer.biases += learning_rate * layer.dbiases;
}
void SGD::post_update_params() {
    iterations += 1;
}