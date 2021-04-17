CXX=g++
OPT=-g -O3 -Wall -Wextra -Wpedantic
CXXFLAGS=$(OPT) -std=c++11

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
