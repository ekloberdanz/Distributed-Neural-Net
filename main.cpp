#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>
# include <mpi.h>

int main() {
    // MPI initialization
    int rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // all
    // Dataset
    Eigen::MatrixXd X_train;
    Eigen::VectorXi y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXi y_test;
 
    // master holds all data
    if (rank == 0) {
        // Load and randomly shuffle the training and testing data from the file
        X_train = load_matrix_data("./data/X_train.csv");
        y_train = load_vector_data("./data/y_train.csv");
        X_test = load_matrix_data("./data/X_test.csv");
        y_test = load_vector_data("./data/y_test.csv");

        int data_total_size = X_train.rows(); // total number of train data points
        int number_of_workers = comm_sz - 1;
        int data_subset_size = data_total_size/number_of_workers;
        int message_size_X = data_subset_size*X_train.cols();
        int message_size_y = data_subset_size

        // send a subset of data to each worker
        for (int dest=1; dest<=number_of_workers; dest++) {
            Eigen::MatrixXd X_train_subset = X_train.block((dest-1)*data_subset_size, 0, dest*data_subset_size, X_train.cols());
            Eigen::VectorXi y_train_subset = y_train.segment((dest-1)*data_subset_size, dest*data_subset_size);

            MPI_Send(message_size_X, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(X_train_subset, message_size_X, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

            MPI_Send(message_size_y, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(y_train_subset, message_size_y, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }

    // workers receive data from master
    if (rank != 0) {
        int source = 0;
        MPI_Recv(message_size_X, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
        MPI_Recv(X_train_subset, message_size_X, MPI_DOUBLE, source, 0, MPI_COMM_WORLD);

        MPI_Recv(message_size_y, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
        MPI_Recv(y_train_subset, message_size_y, MPI_DOUBLE, source, 0, MPI_COMM_WORLD);
    }

    // all
    // Parameters
    int NUM_CLASSES = 3;
    double start_learning_rate = 1.0;

    // all
    // Load initial weights from csv file
    Eigen::MatrixXd w_1;
    Eigen::MatrixXd w_2;
    w_1 = load_matrix_data("./data/weights_1.csv");
    w_2 = load_matrix_data("./data/weights_2.csv");

    // all
    // Create for neural network objects
    LayerDense dense_layer_1(2, 64, w_1);
    ActivationRelu activation_relu;
    LayerDense dense_layer_2(64, NUM_CLASSES, w_2);
    ActivationSoftmax activation_softmax;
    CrossEntropyLoss loss_categorical_crossentropy;
    StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);

    // variables
    double loss;
    double train_accuracy;
    double test_accuracy;
    double pred;
    int index_pred;

    // Train DNN
    int NUMBER_OF_EPOCHS = 1000;
    for (int epoch : boost::irange(0,NUMBER_OF_EPOCHS)) {
        if (rank != 0) {
            ////////////////////////////////////////////////////////forward pass//////////////////////////////////////////////////////////////////////////////////
            dense_layer_1.forward(X_train_subset);
            activation_relu.forward(dense_layer_1.output);
            dense_layer_2.forward(activation_relu.output);
            activation_softmax.forward(dense_layer_2.output);

            ////////////////////////////////////////////////////////////backward pass/////////////////////////////////////////////////////////////////////////
            loss_categorical_crossentropy.backward(activation_softmax.output, y_train_subset);
            activation_softmax.backward(loss_categorical_crossentropy.dinputs);
            dense_layer_2.backward(activation_softmax.dinputs);
            activation_relu.backward(dense_layer_2.dinputs);
            dense_layer_1.backward(activation_relu.dinputs);

            ////////////////////////////////////////////////////////////optimizer - update weights and biases/////////////////////////////////////////////////////////////////////////
            optimizer_SGD.pre_update_params(start_learning_rate);
            optimizer_SGD.update_params(dense_layer_1);
            optimizer_SGD.update_params(dense_layer_2);
            optimizer_SGD.post_update_params();

            // workers send weights and biases to master, who receives them
            // MPI_Sendrecv(dense_layer_1.weights, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, 0, 0, 
            //     dense_layer_1.weights, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Sendrecv(dense_layer_2.weights, dense_layer_2.weights.rows() * dense_layer_2.weights.cols(), MPI_DOUBLE, 0, 0,
            //     dense_layer_2.weights, dense_layer_2.weights.rows() * dense_layer_2.weights.cols(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Sendrecv(dense_layer_1.biases, dense_layer_1.biases.rows() * dense_layer_1.biases.cols(), MPI_DOUBLE, 0, 0, 
            //     dense_layer_1.biases, dense_layer_1.biases.rows() * dense_layer_1.biases.cols(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Sendrecv(dense_layer_2.biases, dense_layer_2.biases.rows() * dense_layer_2.biases.cols(), MPI_DOUBLE, 0, 0,
            //     dense_layer_2.biases, dense_layer_2.biases.rows() * dense_layer_2.biases.cols(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Barrier(MPI_COMM_WORLD); // wait for all workers to compute weights and biases

            // workers send weights and biases to master, who uses them to compute a sum to compute average
            MPI_Reduce(&dense_layer_1.weights, &dense_layer_1.weights, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dense_layer_2.weights, &dense_layer_2.weights, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dense_layer_1.biases, &dense_layer_1.biases, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dense_layer_2.biases, &dense_layer_2.biases, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        if (rank ==0) {
            // compute average
            dense_layer_1.weights = dense_layer_1.weights/number_of_workers;
            dense_layer_2.weights = dense_layer_2.weights/number_of_workers;
            dense_layer_1.biases = dense_layer_1.biases/number_of_workers;
            dense_layer_2.weights = dense_layer_2.weights/number_of_workers;

            // broadcast new weights and biases to workers
            MPI_Bcast(&dense_layer_1.weights, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dense_layer_2.weights, dense_layer_2.weights.rows() * dense_layer_2.weights.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dense_layer_1.biases, dense_layer_1.biases.rows() * dense_layer_1.biases.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dense_layer_2.biases, dense_layer_2.biases.rows() * dense_layer_2.biases.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // periodically calculate train accuracy and loss
            dense_layer_1.forward(X_train);
            activation_relu.forward(dense_layer_1.output);
            dense_layer_2.forward(activation_relu.output);
            activation_softmax.forward(dense_layer_2.output);
            // calculate loss
            loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_train);
            // get predictions and accuracy
            Eigen::MatrixXd::Index maxRow, maxCol;
            Eigen::VectorXi predictions(activation_softmax.output.rows());
            Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

            for (int i=0; i < activation_softmax.output.rows(); i++) {
                pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
                index_pred = maxCol;
                predictions(i) = index_pred;
                pred_truth_comparison(i) = predictions(i) == y_train(i);
            }
            train_accuracy = pred_truth_comparison.mean();
            if (epoch % 100 == 0) {
                std::cout << "epoch: " << epoch << std::endl;
                std::cout << "train_accuracy: " << train_accuracy << std::endl;
                std::cout << "learning_rate: " << optimizer_SGD.learning_rate << std::endl;
                std::cout << "loss: " << loss << std::endl;
            }
        }
    }

    // // Test DNN
    if (rank == 0) {
        dense_layer_1.forward(X_test);
        activation_relu.forward(dense_layer_1.output);
        dense_layer_2.forward(activation_relu.output);
        activation_softmax.forward(dense_layer_2.output);
        // calculate loss
        loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_test);
        // get predictions and accuracy
        Eigen::MatrixXd::Index maxRow, maxCol;
        Eigen::VectorXi predictions(activation_softmax.output.rows());
        Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

        for (int i=0; i < activation_softmax.output.rows(); i++) {
                pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
                index_pred = maxCol;
                predictions(i) = index_pred;
                pred_truth_comparison(i) = predictions(i) == y_test(i);
        }
        test_accuracy = pred_truth_comparison.mean();
        std::cout << "test_accuracy: " << test_accuracy << std::endl;
    }

    MPI_Finalize();
}
