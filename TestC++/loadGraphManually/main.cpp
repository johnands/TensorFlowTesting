#include <iostream>
#include <fstream>
#include <armadillo>
#include <string>
#include <vector>
#include "neuralnetwork.h"
#include "activationfunctions.h"


int main(int numberOfArguments, char *argumentList[]) {

    const char *filename = argumentList[1];

    NeuralNetwork *neuralNetwork = new NeuralNetwork();
    neuralNetwork->readFromFile(filename);

    // generate data
    arma::vec data = arma::linspace(0.9, 1.6, 500);
    //neuralNetwork->network(1.5);

    for (int i=0; i < arma::size(data)[0]; i++) {
        neuralNetwork->network(data(i));
    }

    //std::cout << arma::size(data) << std::endl;


    return 0;
}

