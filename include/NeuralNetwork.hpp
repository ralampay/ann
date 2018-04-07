#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "json.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;
using json = nlohmann::json;

enum ANN_COST {
  COST_MSE
};

enum ANN_ACTIVATION {
  A_TANH,
  A_RELU,
  A_SIGM
};

struct ANNConfig {
  vector<int> topology;
  double bias;
  double learningRate;
  double momentum;
  int epoch;
  ANN_ACTIVATION hActivation;
  ANN_ACTIVATION oActivation;
  ANN_COST cost;
  string trainingFile;
  string labelsFile;
  string weightsFile;
};

class NeuralNetwork
{
public:
  NeuralNetwork(ANNConfig config);

  void train(
        vector<double> input, 
        vector<double> target, 
        double bias, 
        double learningRate, 
        double momentum
      );

  void setCurrentInput(vector<double> input);
  void setCurrentTarget(vector<double> target) { this->target = target; };

  void feedForward();
  void backPropagation();
  void setErrors();

  vector<double> getActivatedVals(int index) { return this->layers.at(index)->getActivatedVals(); }

  Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
  Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
  Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
  Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };

  void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }

  void saveWeights(string file);
  void loadWeights(string file);

  int topologySize;
  int hiddenActivationType  = RELU;
  int outputActivationType  = SIGM;
  int costFunctionType      = COST_MSE;

  vector<int> topology;
  vector<Layer *> layers;
  vector<Matrix *> weightMatrices;
  vector<Matrix *> gradientMatrices;

  vector<double> input;
  vector<double> target;
  vector<double> errors;
  vector<double> derivedErrors;

  double error              = 0;
  double bias               = 1;
  double momentum;
  double learningRate;

  ANNConfig config;

private:
  void setErrorMSE();
};

#endif
