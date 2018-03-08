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
  cout << "weights_generator [configFile]" << endl;
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
  string weightsFile    = config["weightsFile"];

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

  vector< vector<double> > trainingData = utils::Misc::fetchData(trainingFile);
  vector< vector<double> > labelData    = utils::Misc::fetchData(labelsFile);

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
    cout << n->error << endl;

    //cout << "Error at epoch " << i+1 << ": " << n->error << endl;
  }

  cout << "Done! Writing to " << weightsFile << "..." << endl;
  n->saveWeights(weightsFile);

  return 0;
}
