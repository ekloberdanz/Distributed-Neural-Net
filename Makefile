# INCLUDE=-I./include

# neuralnet_serial: main.cpp NeuralNet.hpp NeuralNet.cpp
# 	g++ main.cpp NeuralNet.cpp -o neuralnet_serial -g -O3 $(INCLUDE)

# neuralnet_serial: main.cpp NeuralNet.hpp NeuralNet.cpp
# 	g++ main.cpp NeuralNet.cpp -o neuralnet_serial -g -O3

CXX=g++
OPT=-g -O3 -Wall -Wextra -Wpedantic
CXXFLAGS=$(OPT)

neuralnet: main.o NeuralNet.o
	$(CXX) -o neuralnet main.o NeuralNet.o $(CXXFLAGS)

main.o: main.cpp NeuralNet.hpp
	$(CXX) -o main.o -c main.cpp $(CXXFLAGS)

NeuralNet.o: NeuralNet.hpp NeuralNet.cpp
	$(CXX) -o NeuralNet.o -c NeuralNet.cpp $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f *.o
	rm -f neuralnet
