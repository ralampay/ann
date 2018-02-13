#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::loadWeights(string filename) {
  std::ifstream i(filename);
  json jWeights;
  i >> jWeights;

  vector< vector< vector<double> > > temp = jWeights["weights"];

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    for(int r = 0; r < this->weightMatrices.at(i)->getNumRows(); r++) {
      for(int c = 0; c < this->weightMatrices.at(i)->getNumCols(); c++) {
        this->weightMatrices.at(i)->setValue(r, c, temp.at(i).at(r).at(c));
      }
    }
  }
}
