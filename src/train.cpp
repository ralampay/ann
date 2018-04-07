#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../include/json.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/Misc.hpp"

using namespace std;
using json = nlohmann::json;

void printSyntax() {
  cout << "Syntax:" << endl;
  cout << "train [configFile]" << endl;
}

ANNConfig buildConfig(json configObject) {
  ANNConfig config;

  double learningRate   = configObject["learningRate"];
  double momentum       = configObject["momentum"];
  double bias           = configObject["bias"];
  int epoch             = configObject["epoch"];
  string trainingFile   = configObject["trainingFile"];
  string labelsFile     = configObject["labelsFile"];
  string weightsFile    = configObject["weightsFile"];
  vector<int> topology  = configObject["topology"];

  ANN_ACTIVATION hActivation  = configObject["hActivation"];
  ANN_ACTIVATION oActivation  = configObject["oActivation"];

  config.topology     = topology;
  config.bias         = bias;
  config.learningRate = learningRate;
  config.momentum     = momentum;
  config.epoch        = epoch;
  config.hActivation  = hActivation;
  config.oActivation  = oActivation;
  config.trainingFile = trainingFile;
  config.labelsFile   = labelsFile;
  config.weightsFile  = weightsFile;

  return config;
}

int main(int argc, char **argv) {

  if(argc != 2) {
    printSyntax();
    exit(-1);
  }

  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  NeuralNetwork *n  = new NeuralNetwork(buildConfig(json::parse(str)));

  vector< vector<double> > trainingData = utils::Misc::fetchData(n->config.trainingFile);
  vector< vector<double> > labelData    = utils::Misc::fetchData(n->config.labelsFile);

  cout << "Training Data Size: " << trainingData.size() << endl;
  cout << "Label Data Size: " << labelData.size() << endl;

  for(int i = 0; i < n->config.epoch; i++) {
    for(int tIndex = 0; tIndex < trainingData.size(); tIndex++) {
      vector<double> input    = trainingData.at(tIndex);
      vector<double> target   = labelData.at(tIndex);

      n->train(
        input,
        target,
        n->config.bias,
        n->config.learningRate,
        n->config.momentum
      );
    }
    cout << n->error << endl;

    //cout << "Error at epoch " << i+1 << ": " << n->error << endl;
  }

  cout << "Done! Writing to " << n->config.weightsFile << "..." << endl;
  n->saveWeights(n->config.weightsFile);

  return 0;
}
