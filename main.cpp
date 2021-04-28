#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>
#include <mpi.h>
#include <stdio.h>

int main() {
    // MPI initialization
    int rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    char const* tmp = getenv("TMPDIR");
    std::string TMPDIR(tmp);
    

    // all
    // Dataset
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd X_train_subset;
    Eigen::VectorXi y_train;
    Eigen::VectorXi y_train_subset;
    Eigen::MatrixXd X_test;
    Eigen::VectorXi y_test;
    int data_total_size;
    int data_subset_size;

    // all
    // Parameters
    int NUM_CLASSES = 3;
    double start_learning_rate = 1.0;

    
    // all
    // Load initial weights from csv file
    //Eigen::MatrixXd w_1;
    //Eigen::MatrixXd w_2;
    //w_1 = load_matrix_data("./data/weights_1.csv");
    //w_2 = load_matrix_data("./data/weights_2.csv");

    //std::cout << "w1 : " << w_1.rows() << std::endl;
    
    // all
    // Create for neural network objects
    LayerDense dense_layer_1(2, 64);
    ActivationRelu activation_relu;
    LayerDense dense_layer_2(64, NUM_CLASSES);
    ActivationSoftmax activation_softmax;
    CrossEntropyLoss loss_categorical_crossentropy;
    StochasticGradientDescent optimizer_SGD(1.0, 1e-3, 0.9);

    // variables
    double loss;
    double train_accuracy;
    double test_accuracy;
    double pred;
    int index_pred;

    Eigen::MatrixXd weights_1_sum(dense_layer_1.weights.rows(), dense_layer_1.weights.cols());
    Eigen::MatrixXd weights_2_sum(dense_layer_2.weights.rows(), dense_layer_2.weights.cols());
    Eigen::VectorXd biases_1_sum(dense_layer_1.biases.rows(), dense_layer_1.biases.cols());
    Eigen::VectorXd biases_2_sum(dense_layer_2.biases.rows(), dense_layer_2.biases.cols());
    
    Eigen::MatrixXd weights_2_new(dense_layer_2.weights.rows(), dense_layer_2.weights.cols());
    Eigen::VectorXd biases_1_new(dense_layer_1.biases.rows(), dense_layer_1.biases.cols());
    Eigen::VectorXd biases_2_new(dense_layer_2.biases.rows(), dense_layer_2.biases.cols());
    
    int NUMBER_OF_EPOCHS = 1;
    
    // Load and randomly shuffle the training and testing data from the file
    X_train = load_matrix_data(TMPDIR+"/data/X_train.csv");
    y_train = load_vector_data(TMPDIR+"/data/y_train.csv");
    X_test = load_matrix_data(TMPDIR+"/data/X_test.csv");
    y_test = load_vector_data(TMPDIR+"/data/y_test.csv");
        
    data_total_size = X_train.rows(); // total number of train data points
    //std::cout << "data total rows " << data_total_size << std::endl;

    data_subset_size = data_total_size/(comm_sz-1);
    //std::cout << "data subset size " << data_subset_size << std::endl;
        
    MPI_Barrier(MPI_COMM_WORLD); // wait for all workers to initialize neural net objects

    //std::cout <<"Ready to train" << std::endl;
    
    // Train DNN
    for (int epoch : boost::irange(0,NUMBER_OF_EPOCHS)) {
    //std::cout <<"epoch " << epoch << std::endl;
        if (rank != 0) {
            X_train_subset = X_train.block((rank-1)*data_subset_size, 0, data_subset_size, X_train.cols());
            y_train_subset = y_train.segment((rank-1)*data_subset_size, data_subset_size);
            //std::cout << "The sent matrix X_train_subset is of size " << X_train_subset.rows() << "x" << X_train_subset.cols() << std::endl;
            //std::cout << "The sent vector y_train_subset is of size " << y_train_subset.rows() << "x" << y_train_subset.cols() << std::endl;
            ////////////////////////////////////////////////////////forward pass//////////////////////////////////////////////////////////////////////////////////
            dense_layer_1.forward(X_train_subset);
            activation_relu.forward(dense_layer_1.output);
            dense_layer_2.forward(activation_relu.output);
            activation_softmax.forward(dense_layer_2.output);

            //std::cout <<"Finished forward" << std::endl;
            ////////////////////////////////////////////////////////////backward pass/////////////////////////////////////////////////////////////////////////
            loss_categorical_crossentropy.backward(activation_softmax.output, y_train_subset);
            activation_softmax.backward(loss_categorical_crossentropy.dinputs);
            dense_layer_2.backward(activation_softmax.dinputs);
            activation_relu.backward(dense_layer_2.dinputs);
            dense_layer_1.backward(activation_relu.dinputs);

            //std::cout <<"Finished backward" << std::endl;
            ////////////////////////////////////////////////////////////optimizer - update weights and biases/////////////////////////////////////////////////////////////////////////
            optimizer_SGD.pre_update_params(start_learning_rate);
            optimizer_SGD.update_params(dense_layer_1);
            optimizer_SGD.update_params(dense_layer_2);
            optimizer_SGD.post_update_params();

            //Eigen::MatrixXd dense_layer_1_weights_local = dense_layer_1.weights;
            //std::cout << "weights_1 size " << dense_layer_1.weights.rows() << "x" << dense_layer_1.weights.cols() << "rank: " << rank  << std::endl;
            //std::cout <<"Dense layer weights 1 to send: " << dense_layer_1.weights << std::endl;
            
            MPI_Send(dense_layer_1.weights.data(), 128, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            std::cout <<"sent, rank: " << rank << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD); // wait for all workers to compute weights and biases
        if (rank == 0) {
            std::cout <<"Everyone computed weights and biases" << std::endl;
        }
        
        // workers send weights and biases to master, who uses them to compute a sum to compute average
        //if (rank == 0) {
        //    std::cout <<"critical rank: " << rank << std::endl;
        //    std::cout << "The matrix weights is of size " << dense_layer_1.weights.rows() << "x" << dense_layer_1.weights.cols() << std::endl;
        //    std::cout << "The vector biases is of size " << dense_layer_1.biases.rows() << "x" << dense_layer_1.biases.cols() << std::endl;
        //    weights_1_sum = Eigen::MatrixXd::Zero(dense_layer_1.weights.rows(), dense_layer_1.weights.cols());
        //    std::cout << "The matrix weights sum is of size " << weights_1_sum.rows() << "x" << weights_1_sum.cols() << std::endl;
        //    weights_2_sum = Eigen::MatrixXd::Zero(dense_layer_2.weights.rows(), dense_layer_2.weights.cols());
        //    biases_1_sum = Eigen::VectorXd::Zero(dense_layer_1.biases.rows(), dense_layer_1.biases.cols());
        //    biases_2_sum = Eigen::VectorXd::Zero(dense_layer_2.biases.rows(), dense_layer_2.biases.cols());
        //    std::cout << "The vector biases sum is of size " << biases_1_sum.rows() << "x" << biases_1_sum.cols() << std::endl;
        //    
        //    
        //    for (int worker=1; worker <= comm_sz-1; worker++) {
        //        std::cout <<"worker: " << worker << std::endl;
        //        MPI_Recv(&weights_1_new, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //        std::cout <<"Received new weights from worker: " << worker << std::endl;
        //        std::cout <<"New weights: " << weights_1_new << std::endl;
        //        std::cout <<"Sum weights: " << weights_1_sum << std::endl;
        //        weights_1_sum = weights_1_sum + weights_1_new;
        //        std::cout <<"Updated global sum with weights from worker: " << worker << std::endl;
        //    }

        //std::cout <<" Finished sending and receving " << std::endl;
        
        if (rank == 0) {
            weights_1_sum = Eigen::MatrixXd::Zero(dense_layer_1.weights.rows(), dense_layer_1.weights.cols());
            std::cout << "before weights_1_sum size " << weights_1_sum.rows() << "x" << weights_1_sum.cols() << "rank: " << rank  << std::endl;
            std::cout << "before weights_1_sum " << weights_1_sum << std::endl;
            
            Eigen::MatrixXd weights_1_new;
            for (int p=1; p <= comm_sz-1; p++){
                MPI_Recv(&weights_1_new, 128, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //MPI_Recv(&weights_1_new, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "received, rank: " << rank << std::endl;
                //weights_1_sum = weights_1_sum + new_w_1;    
            }
            std::cout <<"Everyone sent their weights and biases and root collected" << std::endl;
        }

        //MPI_Reduce(&dense_layer_1.weights, &weights_1_sum, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "weights_1_sum size " << weights_1_sum.rows() << "x" << weights_1_sum.cols() << "rank: " << rank  << std::endl;
            std::cout << "weights_1_sum " << weights_1_sum << std::endl;
        }
        //MPI_Reduce(&dense_layer_2.weights, &weights_2_sum, dense_layer_2.weights.rows() * dense_layer_2.weights.cols(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //std::cout <<"We're good! " << std::endl;
        //MPI_Reduce(&dense_layer_1.biases, &biases_1_sum, dense_layer_1.biases.rows() * dense_layer_1.biases.cols(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //std::cout <<"We're good! " << std::endl;
        //MPI_Reduce(&dense_layer_2.biases, &biases_2_sum, dense_layer_2.biases.rows() * dense_layer_2.biases.cols(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //std::cout <<"We're good! " << std::endl;
        
        
        MPI_Barrier(MPI_COMM_WORLD); // wait for all workers to send updates
        if (rank == 0) {
            std::cout <<"Barrier" << std::endl;
        }
        
        //if (rank == 0) {
        //    // compute average
        //    std::cout << "Weights" << dense_layer_1.weights << std::endl;
        //    dense_layer_1.weights = weights_1_sum/(comm_sz-1);
        //    dense_layer_2.weights = weights_2_sum/(comm_sz-1);
        //    dense_layer_1.biases = biases_1_sum/(comm_sz-1);
        //    dense_layer_2.biases = biases_2_sum/(comm_sz-1);

        //    // broadcast new weights and biases to workers
        //    for (int dest=1; dest<comm_sz-1; dest++) {
        //        MPI_Sendrecv_replace(&dense_layer_1.weights, dense_layer_1.weights.rows() * dense_layer_1.weights.cols(), MPI_DOUBLE, dest, 0, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //        MPI_Sendrecv_replace(&dense_layer_2.weights, dense_layer_2.weights.rows() * dense_layer_2.weights.cols(), MPI_DOUBLE, dest, 0, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //        MPI_Sendrecv_replace(&dense_layer_1.biases, dense_layer_1.biases.rows() * dense_layer_1.biases.cols(), MPI_DOUBLE, dest, 0, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //        MPI_Sendrecv_replace(&dense_layer_2.biases, dense_layer_2.biases.rows() * dense_layer_2.biases.cols(), MPI_DOUBLE, dest, 0, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //    }

        //    std::cout <<"Everyone got updated weights and biases" << std::endl;
        //    
        //    // periodically calculate train accuracy and loss
        //    dense_layer_1.forward(X_train);
        //    activation_relu.forward(dense_layer_1.output);
        //    dense_layer_2.forward(activation_relu.output);
        //    activation_softmax.forward(dense_layer_2.output);
        //    loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_train);
        //    Eigen::MatrixXd::Index maxRow, maxCol;
        //    Eigen::VectorXi predictions(activation_softmax.output.rows());
        //    Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

        //    for (int i=0; i < activation_softmax.output.rows(); i++) {
        //        pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
        //        index_pred = maxCol;
        //        predictions(i) = index_pred;
        //        pred_truth_comparison(i) = predictions(i) == y_train(i);
        //    }
        //    train_accuracy = pred_truth_comparison.mean();
        //    if (epoch % 100 == 0) {
        //        std::cout << "epoch: " << epoch << std::endl;
        //        std::cout << "train_accuracy: " << train_accuracy << std::endl;
        //        std::cout << "learning_rate: " << optimizer_SGD.learning_rate << std::endl;
        //        std::cout << "loss: " << loss << std::endl;
        //    }
        //    std::cout <<"Finished epoch: " << epoch << std::endl;
        //}
    }

    // // Test DNN
    //if (rank == 0) {
    //    dense_layer_1.forward(X_test);
    //    activation_relu.forward(dense_layer_1.output);
    //    dense_layer_2.forward(activation_relu.output);
    //    activation_softmax.forward(dense_layer_2.output);
    //    // calculate loss
    //    loss = loss_categorical_crossentropy.calculate(activation_softmax.output, y_test);
    //    // get predictions and accuracy
    //    Eigen::MatrixXd::Index maxRow, maxCol;
    //    Eigen::VectorXi predictions(activation_softmax.output.rows());
    //    Eigen::VectorXd pred_truth_comparison(activation_softmax.output.rows());

    //    for (int i=0; i < activation_softmax.output.rows(); i++) {
    //            pred = activation_softmax.output.row(i).maxCoeff(&maxRow, &maxCol);
    //            index_pred = maxCol;
    //            predictions(i) = index_pred;
    //            pred_truth_comparison(i) = predictions(i) == y_test(i);
    //    }
    //    test_accuracy = pred_truth_comparison.mean();
    //    std::cout << "test_accuracy: " << test_accuracy << std::endl;
    //}

    MPI_Finalize();
    return 0;
}
