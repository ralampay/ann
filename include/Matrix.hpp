#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;

class Matrix
{
public:
  Matrix(int numRows, int numCols, bool isRandom); 

  Matrix *transpose();
  Matrix *copy();

  void setValue(int r, int c, double v) { this->values.at(r).at(c) = v; }
  double getValue(int r, int c) { return this->values.at(r).at(c); }

  vector< vector<double> > getValues() { return this->values; }

  void printToConsole();

  int getNumRows() { return this->numRows; }
  int getNumCols() { return this->numCols; }

private:
  double generateRandomNumber();

  int numRows;
  int numCols;

  vector< vector<double> > values;
};

#endif
