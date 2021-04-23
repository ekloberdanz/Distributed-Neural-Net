#include "NeuralNet.hpp"
#include<fstream>
#include<iostream>
#include <eigen3/Eigen/Dense>

// LayerDense functions definitions
void LayerDense::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    output = (inputs * weights).rowwise() + biases.transpose();
}

void LayerDense::backward(const Eigen::MatrixXd &dvalues) {
    dweights = inputs.transpose() * dvalues;
    dbiases = dvalues.colwise().sum();
    dinputs = dvalues * weights.transpose();
}

// ActivationRelu functions definitions
void ActivationRelu::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    output = (inputs.array() < 0).select(0, inputs);
}

void ActivationRelu::backward(const Eigen::MatrixXd &dvalues) {
    dinputs = (inputs.array() <= 0).select(0, dvalues);
}

// ActivationSoftmax functions definitions
void ActivationSoftmax::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd exp_values;
    Eigen::VectorXd max_values;
    Eigen::MatrixXd probabilities;
    max_values = inputs.rowwise().maxCoeff(); // max
    exp_values = (inputs.colwise() - max_values).array().exp(); // unnormalized probabilities
    Eigen::VectorXd sum_exp = exp_values.rowwise().sum();
    output = (exp_values.array().colwise() / sum_exp.array()); // normalized probabilities
}

void ActivationSoftmax::backward(const Eigen::MatrixXd &dvalues) {
    Eigen::MatrixXd jacobian_matrix;
    Eigen::VectorXd single_output;
    Eigen::VectorXd single_dvalues;
    dinputs = Eigen::MatrixXd:: Zero(dvalues.rows(),dvalues.cols());
    Eigen::MatrixXd single_output_one_hot_encoded;  
    int labels = dvalues.cols();
    for (int i = 0; i < dvalues.rows(); i++) {
        single_output = output.row(i).transpose();
        single_dvalues = dvalues.row(i);
        single_output_one_hot_encoded = single_output.asDiagonal();
        Eigen::MatrixXd dot_product = single_output * single_output.transpose();
        jacobian_matrix = single_output_one_hot_encoded - dot_product;
        Eigen::VectorXd gradient = jacobian_matrix * single_dvalues;
        dinputs.row(i) = gradient;
    }    
}

// CrossEntropyLoss functions definitions
Eigen::VectorXd CrossEntropyLoss::forward(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y_true) {
    int samples = y_true.rows();
    int r;
    int index;
    double conf;
    Eigen::MatrixXd y_pred_clipped;
    Eigen::VectorXd correct_confidences(samples);

    y_pred_clipped = (y_pred.array() < 1e-5).select(1e-5, y_pred);
    y_pred_clipped = (y_pred_clipped.array() > 1 - 1e-5).select(1 - 1e-5, y_pred_clipped);

    for (r=0; r < samples; r++) {
        index = y_true(r);
        conf = y_pred_clipped(r, index);
        correct_confidences(r) = conf;
    }
    Eigen::VectorXd negative_log_likelihoods = - (correct_confidences.array().log());
    return negative_log_likelihoods;
}

void CrossEntropyLoss::backward(const Eigen::MatrixXd &dvalues, const Eigen::VectorXi &y_true) {
    int samples = dvalues.rows();
    int labels = dvalues.cols();
    int index;
    Eigen::MatrixXd y_true_one_hot_encoded;
    y_true_one_hot_encoded = Eigen::MatrixXd::Zero(samples, labels);
    for (int r=0; r < samples; r++) {
        index = y_true(r);
        y_true_one_hot_encoded(r, index) = 1;
    }
    // Calculate gradient
    dinputs = - (y_true_one_hot_encoded.array() / dvalues.array());
    // Normalize gradient
    dinputs = dinputs * (1/double(samples));
}

double CrossEntropyLoss::calculate(const Eigen::MatrixXd &output, const Eigen::VectorXi &y) {
    Eigen::VectorXd sample_losses = this->forward(output, y);
    double data_loss = sample_losses.mean();
    return data_loss;
}


// SGD functions declaration
void StochasticGradientDescent::pre_update_params(double start_learning_rate) {
    if (decay != 0.0) {
        learning_rate = start_learning_rate * (1.0 / (1.0 + (decay * iterations)));
    }
}

void StochasticGradientDescent::update_params(LayerDense &layer) {
    Eigen::MatrixXd weight_updates;
    Eigen::VectorXd bias_updates;
    if (momentum != 0.0) {
        weight_updates = (momentum * layer.weight_momentums) - (learning_rate * layer.dweights);
        layer.weight_momentums = weight_updates;
        bias_updates = (momentum * layer.bias_momentums).array() - (learning_rate * layer.dbiases).array();
        layer.bias_momentums = bias_updates;
    } else {
        weight_updates = -learning_rate * layer.dweights;
        bias_updates = -learning_rate * layer.dbiases;
    }
    layer.weights = layer.weights + weight_updates;
    layer.biases = layer.biases + bias_updates;
}

void StochasticGradientDescent::post_update_params() {
    iterations += 1;
}

Eigen::MatrixXd load_matrix_data(std::string fileToOpen) {
    // REFERENCE: https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
    std::vector<double> matrixEntries;
    std::ifstream matrixDataFile(fileToOpen);
    std::string matrixRowString;
    std::string matrixEntry;
    int matrixRowNumber = 0;
    while (getline(matrixDataFile, matrixRowString)) 
    {
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));
        }
        matrixRowNumber++;
    }
    std::cout << "\nmatrix row number: " << matrixRowNumber << "\n";
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}

Eigen::VectorXi load_vector_data(std::string fileToOpen) {
    // REFERENCE: https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
    
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
    std::vector<int> matrixEntries;
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
    return Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}
