#include "NeuralNet.hpp"
#include<fstream>
#include<iostream>

// LayerDense functions definitions
void LayerDense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    Eigen::MatrixXd output = inputs * weights + biases;
}

void LayerDense::backward(Eigen::MatrixXd dvalues) {
    Eigen::MatrixXd dweights = inputs.transpose() * dvalues;
    Eigen::VectorXd dbiases = dvalues.colwise().sum();
    Eigen::MatrixXd dinputs = weights.transpose() * dvalues;
}

// ActivationRelu functions definitions
void ActivationRelu::forward(Eigen::MatrixXd in) {
    inputs = in;
    output = (inputs.array() < 0).select(0, inputs);
}

void ActivationRelu::backward(Eigen::MatrixXd dvalues) {
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

void Loss::backward(Eigen::MatrixXd dvalues, Eigen::MatrixXd y_true) {
    int samples = dvalues.size();
    Eigen::MatrixXd ones;
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
void StochasticGradientDescent::pre_update_params() {
}

void StochasticGradientDescent::update_params(LayerDense layer) {
    layer.weights += learning_rate * layer.dweights;
    layer.biases += learning_rate * layer.dbiases;
}
void StochasticGradientDescent::post_update_params() {
    iterations += 1;
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