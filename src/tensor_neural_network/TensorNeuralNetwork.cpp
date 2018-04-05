#include "../../include/TensorNeuralNetwork.hpp"

void TensorNeuralNetwork::setCurrentInput(vector<double> data) {
  auto tensorMap = currentInput.tensor<double, 1>();

  for(int i = 0; i < data.size(); i++) {
    tensorMap(i) = data.at(i);
  }
};

void TensorNeuralNetwork::printCurrentInput() {
  int size  = this->config.topology.at(0);
  auto tensorMap = currentInput.tensor<double, 1>();

  cout << "Current Input:\n";
  for(int i = 0; i < size; i++) {
    cout << tensorMap(i);
    if(i != (size - 1)) {
      cout << "\t";
    } else if(i == (size - 1)) {
      cout << endl;
    }
  }
};
