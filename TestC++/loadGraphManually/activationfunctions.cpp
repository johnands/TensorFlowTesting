#include "activationfunctions.h"
#include <cmath>

ActivationFunctions::ActivationFunctions() {

}


arma::mat ActivationFunctions::relu(arma::mat matrix) {

    // loop through vector of Wx + b and apply the relu function
    // i.e. replacing all negative elements with zeros
    for (int i=0; i < arma::size(matrix)[1]; i++) {
        if (matrix(0,i) < 0)
            matrix(0,i) = 0;
    }
    return matrix;
}


arma::mat ActivationFunctions::sigmoid(arma::mat matrix) {

    for (int i=0; i < arma::size(matrix)[1]; i++) {
        double x = matrix(0,i);
        matrix(0,i) = 1./(1 + exp(-x));
    }
    return matrix;
}
