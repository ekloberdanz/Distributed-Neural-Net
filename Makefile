INCLUDE=-I./include

neuralnet_serial: main.cpp NeuralNet.hpp NeuralNet.cpp
	g++ main.cpp NeuralNet.cpp -o neuralnet_serial -g -O3 $(INCLUDE)