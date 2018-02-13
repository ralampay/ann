#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer
{
public:
  Layer(int size);
  Layer(int size, int activationType);
  void setVal(int i, double v);

  Matrix *matrixifyVals();
  Matrix *matrixifyActivatedVals();
  Matrix *matrixifyDerivedVals();

  vector<double> getActivatedVals();

  vector<Neuron *> getNeurons() { return this->neurons; };
  void setNeuron(vector<Neuron *> neurons) { this->neurons = neurons; }
private:
  int size;
  vector<Neuron *> neurons;
};

#endif
