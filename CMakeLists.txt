cmake_minimum_required(VERSION 3.7)
project(ann)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

option(WITH_AUTOENCODER_VISUALIZER "autoencoder visualizer" OFF)

if(WITH_AUTOENCODER_VISUALIZER)
  message("Autoencoder visualizer set. Compiling...")
else()
  message("Autoencoder visualizer will not compile")
endif()

# Tensorflow tensorflow_cc
find_package(TensorflowCC REQUIRED)

add_executable(
  tensor_ann
  src/tensor_ann.cpp
  src/tensor_neural_network/TensorNeuralNetwork.cpp
  src/tensor_neural_network/setup.cpp
)

target_link_libraries(tensor_ann TensorflowCC::Shared)

add_executable(
  autoencoder_train_and_classify
  src/autoencoder_train_and_classify.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/utils/Math.cpp
  src/utils/Misc.cpp
  src/neural_network/NeuralNetwork.cpp
  src/neural_network/setErrors.cpp
  src/neural_network/feedForward.cpp
  src/neural_network/backPropagation.cpp
  src/neural_network/train.cpp
  src/neural_network/loadWeights.cpp
)

add_executable(
  wg
  src/weights_generator.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/utils/Math.cpp
  src/utils/Misc.cpp
  src/neural_network/NeuralNetwork.cpp
  src/neural_network/setErrors.cpp
  src/neural_network/feedForward.cpp
  src/neural_network/backPropagation.cpp
  src/neural_network/train.cpp
)

add_executable(
  train
  src/train.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/utils/Math.cpp
  src/utils/Misc.cpp
  src/neural_network/NeuralNetwork.cpp
  src/neural_network/setErrors.cpp
  src/neural_network/feedForward.cpp
  src/neural_network/backPropagation.cpp
  src/neural_network/train.cpp
)

add_executable(
  autoencoder_classify
  src/autoencoder_classify.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/utils/Math.cpp
  src/utils/Misc.cpp
  src/neural_network/NeuralNetwork.cpp
  src/neural_network/setErrors.cpp
  src/neural_network/feedForward.cpp
  src/neural_network/backPropagation.cpp
  src/neural_network/loadWeights.cpp
  src/neural_network/train.cpp
)
