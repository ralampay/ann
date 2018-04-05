#ifndef _TENSOR_NEURAL_NETWORK_HPP_
#define _TENSOR_NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "json.hpp"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using json = nlohmann::json;

enum TNN_COST {
  MSE
};

enum TNN_ACTIVATION {
  TANH,
  RELU,
  SIGM
};

struct TNNConfig {
  vector<int> topology;
  double bias;
  double learningRate;
  double momentum;
  TNN_ACTIVATION hActivation;
  TNN_ACTIVATION oActivation;
  TNN_COST cost;
};

class TensorNeuralNetwork
{
public:
  TensorNeuralNetwork(TNNConfig config) { 
    this->config = config; 
    this->setup();
  };

  vector<int> getTopology() { return this->config.topology; }

  void setup();
  void setCurrentInput(vector<double> data);
  void printCurrentInput();

private:
  TNNConfig config;

  Tensor currentInput;
  Tensor currentOutput;
  vector<Tensor> hiddenLayers;
  vector<Tensor> weights;
};

#endif
