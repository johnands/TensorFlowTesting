#pragma once
#include <armadillo>

class ActivationFunctions {

public:
    ActivationFunctions();
    static arma::mat relu(arma::mat matrix);
    static arma::mat sigmoid(arma::mat matrix);
};

