#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::setErrors() {
  switch(costFunctionType) {
    case(COST_MSE): this->setErrorMSE(); break;
    default: this->setErrorMSE(); break;
  }
}

void NeuralNetwork::setErrorMSE() {
  int outputLayerIndex            = this->layers.size() - 1;
  vector<Neuron *> outputNeurons  = this->layers.at(outputLayerIndex)->getNeurons();

  this->error = 0.00;

  for(int i = 0; i < target.size(); i++) {
    double t  = target.at(i);
    double y  = outputNeurons.at(i)->getActivatedVal();

    errors.at(i)        = 0.5 * pow(abs((t - y)), 2);
    derivedErrors.at(i) = (y - t);

    this->error += errors.at(i);
  }
}
