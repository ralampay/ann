#include "../include/Layer.hpp"

vector<double> Layer::getActivatedVals() {
  vector<double> ret;

  for(int i = 0; i < this->neurons.size(); i++) {
    double v = this->neurons.at(i)->getActivatedVal();

    ret.push_back(v);
  }

  return ret;
}

void Layer::setVal(int i, double v) {
  this->neurons.at(i)->setVal(v);
}

Layer::Layer(int size) {
  this->size = size;

  for(int i = 0; i < size; i++) {
    Neuron *n = new Neuron(0.000000000);
    this->neurons.push_back(n);
  }
}

Layer::Layer(int size, int activationType) {
  this->size = size;

  for(int i = 0; i < size; i++) {
    Neuron *n = new Neuron(0.000000000, activationType);
    this->neurons.push_back(n);
  }
}

Matrix *Layer::matrixifyVals() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);

  for(int i = 0; i < this->neurons.size(); i++) {
    m->setValue(0, i, this->neurons.at(i)->getVal());
  }

  return m;
}

Matrix *Layer::matrixifyActivatedVals() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);

  for(int i = 0; i < this->neurons.size(); i++) {
    m->setValue(0, i, this->neurons.at(i)->getActivatedVal());
  }

  return m;
}

Matrix *Layer::matrixifyDerivedVals() {
  Matrix *m = new Matrix(1, this->neurons.size(), false);

  for(int i = 0; i < this->neurons.size(); i++) {
    m->setValue(0, i, this->neurons.at(i)->getDerivedVal());
  }

  return m;
}
