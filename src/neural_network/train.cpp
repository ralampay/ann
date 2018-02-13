#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::train(
  vector<double> input, 
  vector<double> target, 
  double bias, 
  double learningRate, 
  double momentum
) {
  this->learningRate  = learningRate;
  this->momentum      = momentum;
  this->bias          = bias;

  this->setCurrentInput(input);
  this->setCurrentTarget(target);

  this->feedForward();
  this->setErrors();
  this->backPropagation();
}
