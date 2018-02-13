#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#define COST_MSE 1

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

class NeuralNetwork
{
public:
  NeuralNetwork(
    vector<int> topology,
    double bias = 1,
    double learningRate = 0.05,
    double momentum = 1
  );

  NeuralNetwork(
    vector<int> topology,
    int hiddenActivationType,
    int outputActivationType,
    int costFunctionType,
    double bias = 1,
    double learningRate = 0.05,
    double momentum = 1
  );

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

private:
  void setErrorMSE();
};

#endif
