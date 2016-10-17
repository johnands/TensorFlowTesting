#pragma once
#include <vector>
#include <armadillo>
#include <fstream>

class NeuralNetwork {

public:
    NeuralNetwork();
    void readFromFile(const char *filename);
    void network(double dataPoint);
    arma::mat relu(arma::mat matrix);

private:
    int m_nLayers;
    int m_nNodes;
    std::vector<arma::mat> m_weights = std::vector<arma::mat>();
    std::vector<arma::mat> m_biases  = std::vector<arma::mat>();
    std::ofstream m_outFile;
};

