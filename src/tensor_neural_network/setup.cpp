#include "../../include/TensorNeuralNetwork.hpp"

void TensorNeuralNetwork::setup() {
  for(int i = 0; i < config.topology.size(); i++) {
    if(i == 0) {
      currentInput  = Tensor(DT_DOUBLE, TensorShape({ config.topology.at(i) }));
    } else if(i == config.topology.at(config.topology.size() - 1)) {
      currentOutput = Tensor(DT_DOUBLE, TensorShape({ config.topology.at(i) }));
    } else {
      hiddenLayers.push_back(
        Tensor(
          DT_DOUBLE,
          TensorShape({ config.topology.at(i) })
        )
      );
    }
  }
};
