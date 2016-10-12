#include "neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <string>

NeuralNetwork::NeuralNetwork() {

}

void NeuralNetwork::readFromFile(const char *filename) {

    std::ifstream input;
    input.open(filename, std::ios::in);

    // check if file successfully opened
    if ( !input.is_open() ) std::cout << "File is not opened" << std::endl;

    // process first line
    std::string activation;
    input >> m_nLayers >> m_nNodes >> activation;
    std::cout << m_nLayers << " " << m_nNodes << " " << activation << std::endl;

    // skip a blank line
    std::string dummyLine;
    std::getline(input, dummyLine);

    // process file
    // store all weights in a temporary vector
    // that will be reshaped later
    std::vector<arma::mat> weightsTemp;
    for ( std::string line; std::getline(input, line); ) {
        //std::cout << line << std::endl;

        if ( line.empty() )
        { std::cout << "yes" << std::endl; break;}


        // store all weights in a vector
        double buffer;                  // have a buffer string
        std::stringstream ss(line);     // insert the string into a stream

        // while there are new weights on current line, add them to vector
        arma::mat matrix(1,m_nNodes);
        int i = 0;
        while ( ss >> buffer ) {
            matrix(0,i) = buffer;
            i++;
        }
        weightsTemp.push_back(matrix);
    }

    // can put all biases in vector directly
    // no need for temporary vector
    for ( std::string line; std::getline(input, line); ) {

        // store all weights in vector
        double buffer;                  // have a buffer string
        std::stringstream ss(line);     // insert the string into a stream

        // while there are new weights on current line, add them to vector
        arma::mat matrix(1,m_nNodes);
        int i = 0;
        while ( ss >> buffer ) {
            matrix(0,i) = buffer;
            i++;
        }
        m_biases.push_back(matrix);
    }

    // write out all weights and biases
    for (const auto i : weightsTemp)
        std::cout << i << std::endl;
    std::cout << std::endl;
    for (const auto i : m_biases)
        std::cout << i << std::endl;

    // resize weights and biases matrices to correct shapes
    m_weights.resize(m_nLayers+1);

    // first hidden layer
    m_weights[0]  = weightsTemp[0];

    // following hidden layers
    for (int i=0; i < m_nLayers-1; i++) {
        m_weights[i+1] = weightsTemp[i*m_nNodes+1];
        for (int j=1; j < m_nNodes; j++) {
            m_weights[i+1] = arma::join_cols(m_weights[i+1], weightsTemp[i+1+j]);
        }
    }

    // output layer
    arma::mat outputLayer = weightsTemp.back();
    m_weights[m_nLayers] = arma::reshape(outputLayer, m_nNodes, 1);

    // reshape bias of output node
    m_biases[m_nLayers].shed_cols(1,m_nNodes-1);

    // write out entire system for comparison
    for (const auto i : m_weights)
        std::cout << i << std::endl;

    for (const auto i : m_biases)
        std::cout << i << std::endl;
}


void NeuralNetwork::network(double dataPoint) {
    // the data needs to be a 1x1 armadillo matrix
    // maybe more than one data point can be processed simultaneously?

    // convert data point to 1x1 matrix
    arma::mat data(1,1); data(0,0) = dataPoint;

    // send data through network
    // use relu as activation except for output layer
    std::vector<arma::mat> activations(m_nLayers+1);
    activations[0] = relu(data*m_weights[0] + m_biases[0]);
    for (int i=0; i < m_nLayers-1; i++) {
        activations[i+1] = relu(activations[i]*m_weights[i+1] + m_biases[i+1]);
    }
    // no activation function for output layer
    activations[m_nLayers] = activations[m_nLayers-1]*m_weights[m_nLayers] + m_biases[m_nLayers];
    std::cout << "OUTPUT: " << activations[m_nLayers] << std::endl;

    //return activations[m_nLayers];
}


arma::mat NeuralNetwork::relu(arma::mat matrix) {

    // loop through vector of Wx + b and apply the relu function
    // i.e. replacing all negative elements with zeros
    for (int i=0; i < arma::size(matrix)[1]; i++) {
        if (matrix(0,i) < 0)
            matrix(0,i) = 0;
    }
    return matrix;
}








