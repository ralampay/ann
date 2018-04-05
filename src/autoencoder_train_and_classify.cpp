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
  cout << "autoencoder_train_and_classify [configFile]" << endl;
}

int main(int argc, char **argv) {

  if(argc != 2) {
    printSyntax();
    exit(-1);
  }

  ifstream configFile(argv[1]);
  string str((std::istreambuf_iterator<char>(configFile)),
              std::istreambuf_iterator<char>());

  auto config = json::parse(str);

  double learningRate   = config["learningRate"];
  double momentum       = config["momentum"];
  double bias           = config["bias"];
  int epoch             = config["epoch"];
  string trainingFile   = config["trainingData"];
  string labelsFile     = config["labelData"];
  string validationFile = config["validationData"];
  string weightsFile    = config["weightsFile"];    // initial weights

  vector<int> topology  = config["topology"];

  cout << "Learning Rate: " << learningRate << endl;
  cout << "Momentum: " << momentum << endl;
  cout << "Bias: " << bias << endl;

  cout << "Topology: " << endl;
  for(int i = 0; i < topology.size(); i++) {
    cout << topology.at(i) << "\t";
  }
  cout << endl;

  NeuralNetwork *n  = new NeuralNetwork(topology, 2, 3, 1, bias, learningRate, momentum);

  cout << "Loading weights from: " << weightsFile << endl;
  n->loadWeights(weightsFile);

  vector< vector<double> > trainingData   = utils::Misc::fetchData(trainingFile);
  vector< vector<double> > labelData      = utils::Misc::fetchData(labelsFile);
  vector< vector<double> > validationData = utils::Misc::fetchData(validationFile);

  cout << "Training Data Size: " << trainingData.size() << endl;
  cout << "Label Data Size: " << labelData.size() << endl;

  for(int i = 0; i < epoch; i++) {
    for(int tIndex = 0; tIndex < trainingData.size(); tIndex++) {
      vector<double> input    = trainingData.at(tIndex);
      vector<double> target   = labelData.at(tIndex);

      n->train(
        input,
        target,
        bias,
        learningRate,
        momentum
      );
    }

    cout << n->error << ",";

    // iterate through validation
    for(int vIndex = 0; vIndex < validationData.size(); vIndex++) {
      vector<double> validation = validationData.at(vIndex);
      n->setCurrentInput(validation);
      n->setCurrentTarget(validation);
      n->feedForward();
      n->setErrors();
      cout << n->error;
      if(vIndex != validationData.size() - 1) {
        cout << ",";
      }
    }
    cout << endl;
  }

  return 0;
}
