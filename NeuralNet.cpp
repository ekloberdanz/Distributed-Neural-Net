#include "NeuralNet.hpp"
#include<fstream>
#include<iostream>
#include <eigen3/Eigen/Dense>

// LayerDense functions definitions
void LayerDense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd output = (inputs * weights).rowwise() + biases.transpose();
    // std::cout <<  typeid(output).name() << std::endl;
    // std::cout << "The matrix output is of size " << output.rows() << "x" << output.cols() << std::endl;
    this->output = output;
    // Eigen::MatrixXd output = inputs * weights;
}

void LayerDense::backward(Eigen::MatrixXd dvalues) {
    Eigen::MatrixXd dweights = inputs.transpose() * dvalues;
    // std::cout << "The matrix dweights is of size " << dweights.rows() << "x" << dweights.cols() << std::endl;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    // std::cout << "The matrix dbiases is of size " << dbiases.rows() << "x" << dbiases.cols() << std::endl;
    Eigen::MatrixXd dinputs = weights * dvalues.transpose();
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    this->dinputs = dinputs;
    this->dweights = dweights;
    this->dbiases = dbiases;
}

// ActivationRelu functions definitions
void ActivationRelu::forward(Eigen::MatrixXd in) {
    inputs = in;
    output = (inputs.array() < 0).select(0, inputs);
    this->output = output;
}

void ActivationRelu::backward(Eigen::MatrixXd dvalues) {
    dinputs = (dvalues.array() <= 0).select(0, dvalues);
    // std::cout << "The matrix in dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    this->dinputs = dinputs;
}

// ActivationSoftmax functions definitions
void ActivationSoftmax::forward(Eigen::MatrixXd in) {
    inputs = in;
    // std::cout << "The matrix inputs is of size " << inputs.rows() << "x" << inputs.cols() << std::endl;
   
    Eigen::MatrixXd exp_values;
    Eigen::VectorXd max_values;
    Eigen::MatrixXd probabilities;

    // max_values.setZero();
    max_values = inputs.rowwise().maxCoeff(); // max
    // std::cout << max_values;
    // std::cout << "The vector max_values is of size " << max_values.rows() << "x" << max_values.cols() << std::endl;
    exp_values = (inputs.colwise() - max_values).array().exp(); // unnormalized probabilities
    // std::cout << "The matrix exp_values is of size " << exp_values.rows() << "x" << exp_values.cols() << std::endl;
    // std::cout << "The vector exp_values.rowwise().sum() is of size " << exp_values.rowwise().sum().rows() << "x" << exp_values.rowwise().sum().cols() << std::endl;
    probabilities = exp_values.transpose() * (exp_values.rowwise().sum().asDiagonal().inverse()); // normalized probabilities
    output = probabilities;
    this->output = output;
}

void ActivationSoftmax::backward(Eigen::MatrixXd dvalues) {
    Eigen::MatrixXd jacobian_matrix;
    Eigen::MatrixXd single_output;
    Eigen::VectorXd single_dvalues;
    Eigen::MatrixXd dinputs(dvalues.rows(),dvalues.cols());
    Eigen::MatrixXd single_output_one_hot_encoded;  
    int labels = dvalues.cols();

    //for (i, (single_output, single_dvalues) in enum boost::combine(output, dvalues); i++) {
    for (int i = 0; i < dvalues.rows(); i++)    {
        single_output = output.col(i);
        // std::cout << "The vector single_output is of size " << single_output.rows() << "x" << single_output.cols() << std::endl;
        single_dvalues = dvalues.row(i);
        // std::cout << "The vector single_dvalues is of size " << single_dvalues.rows() << "x" << single_dvalues.cols() << std::endl;
        single_output_one_hot_encoded = Eigen::MatrixXd::Zero(labels, labels);
        for (int r=0; r < labels; r++) {
            single_output_one_hot_encoded(r, r) = single_output(r);
        }
        // std::cout << "The matrix single_output_one_hot_encoded is of size " << single_output_one_hot_encoded.rows() << "x" << single_output_one_hot_encoded.cols() << std::endl;
        // std::cout << "single_output_one_hot_encoded " << single_output_one_hot_encoded;
        jacobian_matrix = single_output_one_hot_encoded - (single_output * single_output.transpose());
        // std::cout << "The matrix jacobian_matrix is of size " << jacobian_matrix.rows() << "x" << jacobian_matrix.cols() << std::endl;
        // std::cout << "The vector single_dvalues is of size " << single_dvalues.rows() << "x" << single_dvalues.cols() << std::endl;
        // std::cout << "The vector dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
        Eigen::VectorXd gradient = jacobian_matrix * single_dvalues;
        // std::cout << "The vector gradient is of size " << gradient.rows() << "x" << gradient.cols() << std::endl;
        for (int col=0; col < dinputs.cols(); col++) {
            dinputs(i, col) = gradient(col);
            }
        }    
    this->dinputs = dinputs;
}

// Loss functions definitions
Eigen::VectorXd Loss::forward(Eigen::MatrixXd y_pred, Eigen::VectorXd y_true) {
    int samples = y_true.rows();
    
    // std::cout << "The matrix y_pred is of size " << y_pred.rows() << "x" << y_pred.cols() << std::endl;
    // std::cout << "The vector y_true is of size " << y_true.rows() << "x" << y_true.cols() << std::endl;
    int r;
    int index;
    double conf;
    Eigen::VectorXd correct_confidences(samples);
    for (r=0; r < samples; r++) {
        // std::cout << "r: " << r << std::endl;
        index = y_true(r);
        // std::cout << "index: " << index << std::endl;
        conf = y_pred(index, r);
        // std::cout << "confidence: " << conf << std::endl;
        correct_confidences(r) = conf;
    }
    // std::cout << "samples: " << samples << std::endl;
    // std::cout << "The vector correct_confidences is of size " << correct_confidences.rows() << "x" << correct_confidences.cols() << std::endl;
    Eigen::VectorXd negative_log_likelihoods = correct_confidences.array().log();
    return negative_log_likelihoods;
}

void Loss::backward(Eigen::MatrixXd dvalues, Eigen::VectorXd y_true) {
    int samples = dvalues.rows();
    int labels = dvalues.cols();
    int index;
    Eigen::MatrixXd y_true_one_hot_encoded;
    y_true_one_hot_encoded = Eigen::MatrixXd::Zero(samples, labels);
    for (int r=0; r < samples; r++) {
        index = y_true(r);
        y_true_one_hot_encoded(r, index) = 1;
    }
    // std::cout << "y_true_one_hot_encoded " << y_true_one_hot_encoded;
    // std::cout << "The matrix y_true_one_hot_encoded is of size " << y_true_one_hot_encoded.rows() << "x" << y_true_one_hot_encoded.cols() << std::endl;
    // Calculate gradient
    dinputs = - y_true_one_hot_encoded.array() / dvalues.array();
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    // Normalize gradient
    dinputs = dinputs/double(samples);
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    this->dinputs = dinputs;
}

double Loss::calculate(Eigen::MatrixXd output, Eigen::VectorXd y) {
    Eigen::VectorXd sample_losses = Loss::forward(output, y);
    double data_loss = sample_losses.mean();
    return data_loss;
}


// SGD functions declaration
void StochasticGradientDescent::pre_update_params(double start_learning_rate) {
    if (decay != 0.0) {
        // std::cout << "yes" << std::endl;
        // this->learning_rate = learning_rate;
        learning_rate = start_learning_rate * (1.0 / (1.0 + (decay * iterations)));
        // this->learning_rate = learning_rate;
        // std::cout << "learning_rate:  " << learning_rate << std::endl;
        // std::cout << "iterations:  " << iterations << std::endl;

        // std::cout << "iterations:  " << iterations << std::endl;
    }
}

void StochasticGradientDescent::update_params(LayerDense &layer) {
    // std::cout << "The matrix optimizer layer.weights is of size " << layer.weights.rows() << "x" << layer.weights.cols() << std::endl;
    // std::cout << "The matrix optimizer layer.dweights is of size " << layer.dweights.rows() << "x" << layer.dweights.cols() << std::endl;
    // std::cout << "The matrix optimizer layer.biases is of size " << layer.biases.rows() << "x" << layer.biases.cols() << std::endl;
    // std::cout << "The matrix optimizer layer.dbiases is of size " << layer.dbiases.rows() << "x" << layer.dbiases.cols() << std::endl;
    
    Eigen::MatrixXd weight_updates;
    Eigen::VectorXd bias_updates;

    if (momentum != 0.0) {
        weight_updates = momentum * layer.weight_momentums - learning_rate * layer.dweights;
        layer.weight_momentums = weight_updates;
        bias_updates = momentum * layer.bias_momentums - learning_rate * layer.dbiases;
        layer.bias_momentums = bias_updates;
    } else {
        // std::cout << "here" << std::endl;
        // std::cout << "learning_rate here:  " << learning_rate << std::endl;
        weight_updates = - learning_rate * layer.dweights;
        bias_updates =- learning_rate * layer.dbiases;
    }
    layer.weights += weight_updates;
    layer.biases += bias_updates;

    layer.weights = layer.weights;
    layer.biases  = layer.biases;

    // layer.weights += (-layer.dweights * learning_rate);
    // // std::cout << "The matrix layer.weights is of size " << layer.weights.rows() << "x" << layer.weights.cols() << std::endl;
    // layer.biases += (-learning_rate * layer.dbiases);
    // // std::cout << "The matrix layer.biases is of size " << layer.biases.rows() << "x" << layer.biases.cols() << std::endl;
}

void StochasticGradientDescent::post_update_params() {
    iterations += 1;
    this->iterations = iterations;
}

Eigen::MatrixXd load_matrix_data(std::string fileToOpen) {
    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format
    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c 
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;
    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);
    // this variable is used to store the row of the matrix that contains commas 
    std::string matrixRowString;
    // this variable is used to store the matrix entry;
    std::string matrixEntry;
    // this variable is used to track the number of rows
    int matrixRowNumber = 0;
    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }
    // here we convet the vector variable into the matrix and return the resulting object, 
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}

Eigen::VectorXd load_vector_data(std::string fileToOpen) {
    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format
    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c 
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;
    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);
    // this variable is used to store the row of the matrix that contains commas 
    std::string matrixRowString;
    // this variable is used to store the matrix entry;
    std::string matrixEntry;
    // this variable is used to track the number of rows
    int matrixRowNumber = 0;
    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }
    // here we convet the vector variable into the matrix and return the resulting object, 
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}