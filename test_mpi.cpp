#include "NeuralNet.hpp" 
#include <string>
#include <fstream>
#include <iostream>
#include <boost/range/irange.hpp>
#include <typeinfo>
#include <mpi.h>

int main(int argc, char *argv[]) {
    // MPI initialization
    int rank, comm_sz;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
   // if (rank==0) {
     //   std::cout << "number of processors: " << comm_sz << std::endl;
   // }
    printf( "Hello world from process %d of %d\n", rank, comm_sz);
    //std::cout << "number of processors: " << comm_sz << std::endl;
    MPI_Finalize();
    return 0;
}
