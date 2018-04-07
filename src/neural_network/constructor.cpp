#include "../../include/NeuralNetwork.hpp"

// Constructor
NeuralNetwork::NeuralNetwork(ANNConfig config) {
  this->config        = config;
  this->topology      = config.topology;
  this->topologySize  = config.topology.size();
  this->learningRate  = config.learningRate;
  this->momentum      = config.momentum;
  this->bias          = config.bias;

  this->hiddenActivationType  = config.hActivation;
  this->outputActivationType  = config.oActivation;
  this->costFunctionType      = config.cost;

  for(int i = 0; i < topologySize; i++) {
    if(i > 0 && i < (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
      this->layers.push_back(l);
    } else if(i == (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->outputActivationType);
      this->layers.push_back(l);
    } else {
      Layer *l  = new Layer(topology.at(i));
      this->layers.push_back(l);
    }
  }

  for(int i = 0; i < (topologySize - 1); i++) {
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
    derivedErrors.push_back(0.00);
  }

  this->error = 0.00;
}
