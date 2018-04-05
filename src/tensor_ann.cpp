#include <iostream>

#include "../include/TensorNeuralNetwork.hpp"

using namespace std;

int main() {
  vector<double> test;
  test.push_back(1.1);
  test.push_back(2.2);
  test.push_back(3.3);

  vector<int> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);

  TNNConfig config;
  config.topology     = topology;
  config.bias         = 1;
  config.learningRate = 0.05;
  config.momentum     = 0.09;
  config.hActivation  = RELU;
  config.oActivation  = SIGM;
  config.cost         = MSE;

  TensorNeuralNetwork *nn = new TensorNeuralNetwork(config);
  nn->setCurrentInput(test);
  nn->printCurrentInput();
  return 0;
}
