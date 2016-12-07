#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "math/random.h"
#include "/home/johnands/Documents/FYS4460-MD/math/activationfunctions.h"
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <armadillo>

using namespace tensorflow;

int speedTensorflow();
void speedArmadillo();

int main(int argc, char* argv[]) {

    //int returning = speedTensorflow();
    speedArmadillo();
}

int speedTensorflow() {

    std::ofstream outFile;
    outFile.open("../speedC++.dat", std::ofstream::trunc);
    outFile << "nLayers " << "nNodes " << "Time" << std::endl;
    outFile.close();

    // read and evalute network for different architectures
    int maxLayers = 30;
    int maxNodes = 100;
    for (int layers=1; layers < maxLayers+1; layers++) {
        for (int nodes= 1; nodes < maxNodes+1; nodes++) {

            // construct graph name
            std::string graphName = "../Graphs/frozen_graph";
            graphName.append("L");
            graphName.append(std::to_string(layers));
            graphName.append("N");
            graphName.append(std::to_string(nodes));
            graphName.append(".pb");

            Session* session;
            Status status = NewSession(SessionOptions(), &session);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                return 1;
            }

            GraphDef graph_def;
            status = ReadBinaryProto(Env::Default(), graphName, &graph_def);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                return 1;
            }

            // Add the graph to the session
            status = session->Create(graph_def);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                return 1;
            }

            // Setup inputs and outputs:
            double a = 0.8; double b = 2.5;

            int N = 50;
            double timeElapsed = 0;
            for (int i=0; i < N; i++) {

                double distance = Random::nextDouble()*(b - a) + a;

                Tensor r_ij(DT_FLOAT, TensorShape({1,1}));
                r_ij.scalar<float>()() = distance;

                std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                    { "input/x-input", r_ij},
                };

                // the session will initialize the outputs
                std::vector<tensorflow::Tensor> outputs;

                // evalute network and record time usage
                clock_t start, finish;
                start = clock();
                status = session->Run(inputs, {"outputLayer/activation"}, {}, &outputs);
                finish = clock();

                timeElapsed += double(finish - start) / CLOCKS_PER_SEC;
            }

            outFile.open("../speedC++.dat", std::ofstream::app);
            outFile << layers << " " << nodes << " " << timeElapsed / (double) N << std::endl;
            outFile.close();

            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                return 1;
            }

            // Free any resources used by the session
            session->Close();
        }
    }

    return 0;
}


void speedArmadillo() {

    std::ofstream outFile;
    outFile.open("../speedArmadillo.dat", std::ofstream::trunc);
    outFile << "nLayers " << "nNodes " << "Time" << std::endl;
    outFile.close();

    // read and evalute network for different architectures
    int maxLayers = 30;
    int maxNodes = 100;
    for (int layers=1; layers < maxLayers+1; layers++) {
        for (int nodes= 1; nodes < maxNodes+1; nodes++) {

            // construct graph name
            std::string graphName = "../ManualGraphs/graph";
            graphName.append("L");
            graphName.append(std::to_string(layers));
            graphName.append("N");
            graphName.append(std::to_string(nodes));
            graphName.append(".dat");

            /* ------------------------------------------------
            read graph
            --------------------------------------------------*/

            std::ifstream input;
            input.open(graphName, std::ios::in);

            // check if file successfully opened
            if ( !input.is_open() ) std::cout << "File is not opened" << std::endl;

            // process first line
            std::string activation;
            int m_nLayers, m_nNodes;
            input >> m_nLayers >> m_nNodes >> activation;
            std::cout << m_nLayers << " " << m_nNodes << " " << activation << std::endl;

            // set sizes
            std::vector<arma::mat> weights(layers+1);
            std::vector<arma::mat> weightsTransposed(layers+1);
            std::vector<arma::mat> biases;
            std::vector<arma::mat> preActivations(layers+2);
            std::vector<arma::mat> activations(layers+2);

            // skip a blank line
            std::string dummyLine;
            std::getline(input, dummyLine);

            // process file
            // store all weights in a temporary vector
            // that will be reshaped later
            std::vector<arma::mat> weightsTemp;
            for ( std::string line; std::getline(input, line); ) {

                if ( line.empty() ) break;

                // store all weights in a vector
                double buffer;                  // have a buffer string
                std::stringstream ss(line);     // insert the string into a stream

                // while there are new weights on current line, add them to vector
                arma::mat matrix(1,nodes);
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
                arma::mat matrix(1,nodes);
                int i = 0;
                while ( ss >> buffer ) {
                    matrix(0,i) = buffer;
                    i++;
                }
                biases.push_back(matrix);
            }

            /* ------------------------------------------------
            construct network
            --------------------------------------------------*/

            // first hidden layer
            weights[0]  = weightsTemp[0];

            // following hidden layers
            for (int i=0; i < layers-1; i++) {
                weights[i+1] = weightsTemp[i*nodes+1];
                for (int j=1; j < nodes; j++) {
                    weights[i+1] = arma::join_cols(weights[i+1], weightsTemp[i*nodes+1+j]);
                }
            }

            // output layer
            arma::mat outputLayer = weightsTemp.back();
            weights[layers] = arma::reshape(outputLayer, nodes, 1);

            // reshape bias of output node
            biases[layers].shed_cols(1,nodes-1);

            weightsTransposed.resize(layers+1);
            // obtained transposed matrices
            for (int i=0; i < weights.size(); i++)
                weightsTransposed[i] = weights[i].t();

            /* ------------------------------------------------
            evaluate network
            --------------------------------------------------*/

            // generate data point
            double a = 0.8; double b = 2.5;

            int N = 50;
            double timeElapsed = 0;
            for (int i=0; i < N; i++) {
                double dataPoint = Random::nextDouble()*(b - a) + a;

                // record time usage
                clock_t start, finish;
                start = clock();

                arma::mat data(1,1); data(0,0) = dataPoint;

                preActivations[0] = data;
                activations[0] = preActivations[0];

                for (int i=0; i < layers; i++) {
                    preActivations[i+1] = activations[i]*weights[i] + biases[i];
                    activations[i+1] = ActivationFunctions::sigmoid(preActivations[i+1]);
                }

                preActivations[layers+1] = activations[layers]*weights[layers] + biases[layers];
                activations[layers+1] = preActivations[layers+1];
                finish = clock();

                timeElapsed += double(finish - start) / CLOCKS_PER_SEC;
            }

            // write to file
            outFile.open("../speedArmadillo.dat", std::ofstream::app);
            outFile << layers << " " << nodes << " " << timeElapsed / (double) N << std::endl;
            outFile.close();
        }
    }
}







