#include "NeuralNet.hpp"
#include<fstream>
#include<iostream>
#include <eigen3/Eigen/Dense>

// LayerDense functions definitions
void LayerDense::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    // std::cout << "The matrix biases is of size " << biases.rows() << "x" << biases.cols() << std::endl;
    // std::cout << "The matrix inputs is of size " << inputs.rows() << "x" << inputs.cols() << std::endl;
    // std::cout << "The matrix weights is of size " << weights.rows() << "x" << weights.cols() << std::endl;

    // std::cout << "The matrix (inputs * weights) is of size " << (inputs * weights).rows() << "x" << (inputs * weights).cols() << std::endl;
    output = (inputs * weights).rowwise() + biases.transpose();
    // std::cout << "The matrix output is of size " << output.rows() << "x" << output.cols() << std::endl;
    // std::cout <<  typeid(output).name() << std::endl;
    // std::cout << "The matrix output is of size " << output.rows() << "x" << output.cols() << std::endl;
    // this->output = output;
    // Eigen::MatrixXd output = inputs * weights;
    // std::cout << "inputs in forward: " << inputs<< std::endl;
}

void LayerDense::backward(const Eigen::MatrixXd &dvalues) {
    // std::cout << "The matrix inputs is of size " << inputs.rows() << "x" << inputs.cols() << std::endl;
    // std::cout << "The matrix dvalues is of size " << dvalues.rows() << "x" << dvalues.cols() << std::endl;
    // std::cout << "dvalues: " << dvalues.mean() << std::endl;
    // std::cout << "inputs: " << inputs.mean() << std::endl;

    // std::cout << "The matrix output is of size " << inputs.rows() << "x" << inputs.cols() << std::endl;
    // std::cout << "The matrix output is of size " << inputs.transpose().rows() << "x" << inputs.transpose().cols() << std::endl;
    dweights = inputs.transpose() * dvalues;
    // std::cout << "inputs: " << inputs.mean() << std::endl;
    // std::cout << "The matrix dvalues is of size " << dvalues.rows() << "x" << dvalues.cols() << std::endl;
    // std::cout << "The matrix dweights is of size " << dweights.rows() << "x" << dweights.cols() << std::endl;
    dbiases = dvalues.colwise().sum();
    // std::cout << "The matrix dbiases is of size " << dbiases.rows() << "x" << dbiases.cols() << std::endl;
    dinputs = weights * dvalues.transpose();
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    // std::cout << "inputs in backward: " << inputs.mean() << std::endl;


    // std::cout << "this: " << typeid(this).name() << std::endl;
    // std::cout << "here dbiases: " << dbiases << std::endl;
    // std::cout << "here biases: " << biases << std::endl;


}

// ActivationRelu functions definitions
void ActivationRelu::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    output = (inputs.array() < 0).select(0, inputs);
    // std::cout << "output " << output << std::endl;
    // this->output = output;
}

void ActivationRelu::backward(const Eigen::MatrixXd &dvalues) {
    // dinputs = dvalues;
    // std::cout << "dinputs" << dinputs.mean() << std::endl;
    // std::cout << "dvalues" << dvalues.mean() << std::endl;
    dinputs = (inputs.array() <= 0).select(0, dvalues);
    // std::cout << "The matrix in dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    // std::cout << "dinputs" << dinputs << std::endl;

}

// ActivationSoftmax functions definitions
void ActivationSoftmax::forward(const Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    // std::cout << "The matrix inputs is of size " << inputs.rows() << "x" << inputs.cols() << std::endl;
   
    Eigen::MatrixXd exp_values;
    Eigen::VectorXd max_values;
    Eigen::MatrixXd probabilities;

    this->inputs = inputs;
    // max_values.setZero();
    max_values = inputs.rowwise().maxCoeff(); // max
    // std::cout << "The vector max_values is of size " << max_values.rows() << "x" << max_values.cols() << std::endl;
    // std::cout << max_values;
    exp_values = (inputs.colwise() - max_values).array().exp(); // unnormalized probabilities
    // std::cout << "The matrix exp_values is of size " << exp_values.rows() << "x" << exp_values.cols() << std::endl;
    // std::cout << "The vector exp_values.rowwise().sum() is of size " << exp_values.rowwise().sum().rows() << "x" << exp_values.rowwise().sum().cols() << std::endl;
    // probabilities = exp_values.transpose() * (exp_values.rowwise().sum().asDiagonal().inverse()); // normalized probabilities
    // std::cout << "The vector exp_values.transpose().array() is of size " << exp_values.transpose().array().rows() << "x" << exp_values.transpose().array().cols() << std::endl;
    Eigen::VectorXd sum_exp = exp_values.rowwise().sum();
    // std::cout << "The vector sum_exp is of size " << sum_exp.rows() << "x" << sum_exp.cols() << std::endl;
    output = (exp_values.array().colwise() / sum_exp.array()); // normalized probabilities
    // std::cout << "The vector probabilities is of size " << probabilities.rows() << "x" << probabilities.cols() << std::endl;
    // output.transpose();
    // std::cout << " softmax forward output" << output.mean() << std::endl;
    // std::cout << " softmax forward inputs" << inputs.mean() << std::endl;
    // std::cout << output;
    // this->output = output;
    
}

void ActivationSoftmax::backward(const Eigen::MatrixXd &dvalues) {
    Eigen::MatrixXd jacobian_matrix;
    Eigen::MatrixXd single_output;
    Eigen::VectorXd single_dvalues;
    // Eigen::MatrixXd dinputs(dvalues.rows(),dvalues.cols());
    dinputs = Eigen::MatrixXd:: Zero(dvalues.rows(),dvalues.cols());
    Eigen::MatrixXd single_output_one_hot_encoded;  
    int labels = dvalues.cols();
    // std::cout << "labels " << labels << std::endl;;


    //for (i, (single_output, single_dvalues) in enum boost::combine(output, dvalues); i++) {
    for (int i = 0; i < dvalues.rows(); i++) {
        single_output = output.row(i);
        // std::cout << "The vector single_output is of size " << single_output.rows() << "x" << single_output.cols() << std::endl;
        // std::cout << "single_output " << single_output << std::endl;
        single_dvalues = dvalues.row(i);
        // std::cout << "The vector single_dvalues is of size " << single_dvalues.rows() << "x" << single_dvalues.cols() << std::endl;
        single_output_one_hot_encoded = single_dvalues.asDiagonal();
        // std::cout << "single_output_one_hot_encoded " << single_output_one_hot_encoded << std::endl;
        // single_output_one_hot_encoded = Eigen::MatrixXd::Zero(labels, labels);
        // for (int r=0; r < labels-1; r++) {
        //     std::cout << "\n" << "r " << r << std::endl;;
        //     single_output_one_hot_encoded(r, r) = single_output(r);
        //     std::cout << "single_output_one_hot_encoded " << "\n" << single_output_one_hot_encoded;

        // }
        // std::cout << "The matrix single_output is of size " << single_output.rows() << "x" << single_output.cols() << std::endl;
        // std::cout << "single_output.transpose() * single_output\n " << single_output.transpose() * single_output  << std::endl;
        jacobian_matrix = single_output_one_hot_encoded - (single_output.transpose() * single_output);
        // std::cout << "\nThe matrix jacobian_matrix is of size " << jacobian_matrix.rows() << "x" << jacobian_matrix.cols() << std::endl;
        // std::cout << "jacobian_matrix " << jacobian_matrix << std::endl;
        // std::cout << "The vector dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
        Eigen::VectorXd gradient = jacobian_matrix * single_dvalues;
        // std::cout << "The vector gradient is of size " << gradient.rows() << "x" << gradient.cols() << std::endl;
        // std::cout << "gradient " << gradient << std::endl;
        // std::cout << "The vector dvalues is of size " << dvalues.rows() << "x" << dvalues.cols() << std::endl;
        // dinputs = Eigen::MatrixXd:: Zero(dvalues.rows(),dvalues.cols());
        dinputs.row(i) = gradient;
        // for (int col=0; col < dvalues.cols(); col++) {
        //     std::cout << col << std::endl;
        //     dinputs(i, col) = gradient(col);
        // }
        // std::cout << jacobian_matrix << std::endl;
        // std::cout << single_dvalues << std::endl;
    }    
    // std::cout << "dinputs" << dinputs << std::endl;
    // std::cout << "The mean is: " << dinputs.mean() << std::endl;    
    // std::cout << "The mean is: " << dvalues.mean() << std::endl;  
    // this->dinputs = dinputs;
}

// CrossEntropyLoss functions definitions
Eigen::VectorXd CrossEntropyLoss::forward(const Eigen::MatrixXd &y_pred, const Eigen::VectorXd &y_true) {
    // std::cout << "The matrix y_pred is of size " << y_pred.rows() << "x" << y_pred.cols() << std::endl;
    // std::cout << "The vector y_true is of size " << y_true.rows() << "x" << y_true.cols() << std::endl;
   
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
        conf = y_pred(r, index);
        // std::cout << "confidence: " << conf << std::endl;
        correct_confidences(r) = conf;
    }
    // std::cout << "y_pred: " << y_pred << std::endl;
    // std::cout << "correct_confidences: " << correct_confidences << std::endl;
    // std::cout << "The vector correct_confidences is of size " << correct_confidences.rows() << "x" << correct_confidences.cols() << std::endl;
    Eigen::VectorXd negative_log_likelihoods = - (correct_confidences.array().log());
    // std::cout << "negative_log_likelihoods: " << negative_log_likelihoods << std::endl;
    return negative_log_likelihoods;
}

void CrossEntropyLoss::backward(const Eigen::MatrixXd &dvalues, const Eigen::VectorXd &y_true) {
    // std::cout << "The matrix dvalues is of size " << dvalues.rows() << "x" << dvalues.cols() << std::endl;
    // std::cout << "The vector y_true is of size " << y_true.rows() << "x" << y_true.cols() << std::endl;
    int samples = dvalues.cols();
    int labels = dvalues.rows();
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
    dinputs = - (y_true_one_hot_encoded.array() / dvalues.transpose().array());
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    // Normalize gradient
    dinputs = dinputs/double(samples);
    // std::cout << "The matrix dinputs is of size " << dinputs.rows() << "x" << dinputs.cols() << std::endl;
    // this->dinputs = dinputs;
    // std::cout << "loss_categorical_crossentropy dinputs" << dinputs.mean() << std::endl;
    // std::cout << "loss_categorical_crossentropy dvalues" << dvalues.mean() << std::endl;

}

double CrossEntropyLoss::calculate(const Eigen::MatrixXd &output, const Eigen::VectorXd &y) {
    Eigen::VectorXd sample_losses = this->forward(output, y);
    double data_loss = sample_losses.mean();
    // std::cout << "output: " << output << std::endl;    
    // std::cout << "y: " << y << std::endl;
    // std::cout << "sample_losses: " << sample_losses << std::endl;
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
        weight_updates = -learning_rate * layer.dweights;
        bias_updates = -learning_rate * layer.dbiases;
        // std::cout << "here" << std::endl;
    }
    // std::cout << "The matrix layer.weights is of size " << layer.weights.rows() << "x" << layer.weights.cols() << std::endl;
    // std::cout << "The matrix layer.dweights is of size " << layer.dweights.rows() << "x" << layer.dweights.cols() << std::endl;
    // std::cout << "The matrix weight_updates is of size " << weight_updates.rows() << "x" << weight_updates.cols() << std::endl;
    layer.weights += weight_updates;
    layer.biases += bias_updates;

    // std::cout << "inside weights " << layer.weights.mean() << std::endl;
    std::cout << "inside derviv weights " << layer.dweights.mean() << std::endl;
    std::cout << "inside weight_updates " << weight_updates.mean() << std::endl;
    // std::cout << "inside biases " << layer.biases.mean() << std::endl;
    // std::cout << "inside dbiases " << layer.dbiases.mean() << std::endl;


    // layer.weights = layer.weights;
    // layer.biases  = layer.biases;

    // layer.weights += (-layer.dweights * learning_rate);
    // // std::cout << "The matrix layer.weights is of size " << layer.weights.rows() << "x" << layer.weights.cols() << std::endl;
    // layer.biases += (-learning_rate * layer.dbiases);
    // // std::cout << "The matrix layer.biases is of size " << layer.biases.rows() << "x" << layer.biases.cols() << std::endl;
}

void StochasticGradientDescent::post_update_params() {
    iterations += 1;
    // this->iterations = iterations;
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
