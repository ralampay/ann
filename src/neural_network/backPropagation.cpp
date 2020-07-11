#include "../../include/NeuralNetwork.hpp"
#include "../../include/utils/Math.hpp"

void NeuralNetwork::backPropagation() {
  vector<Matrix *> newWeights;
  Matrix *deltaWeights;
  Matrix *gradients;
  Matrix *derivedValues;
  Matrix *gradientsTransposed;
  Matrix *zActivatedVals;
  Matrix *tempNewWeights;
  Matrix *pGradients;
  Matrix *transposedPWeights;
  Matrix *hiddenDerived;
  Matrix *transposedHidden;

  /**
   *  PART 1: OUTPUT TO LAST HIDDEN LAYER
   */
  int indexOutputLayer  = this->topology.size() - 1;

  gradients = new Matrix(
                1,
                this->topology.at(indexOutputLayer),
                false
              );

  derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();

  for(int i = 0; i < this->topology.at(indexOutputLayer); i++) {
    double e  = this->derivedErrors.at(i);
    double y  = derivedValues->getValue(0, i);
    double g  = e * y;
    gradients->setValue(0, i, g);
  }

  // Gt * Z
  gradientsTransposed = gradients->transpose();
  zActivatedVals      = this->layers.at(indexOutputLayer - 1)->matrixifyActivatedVals();

  deltaWeights  = new Matrix(
                    gradientsTransposed->getNumRows(),
                    zActivatedVals->getNumCols(),
                    false
                  );

  ::utils::Math::multiplyMatrix(gradientsTransposed, zActivatedVals, deltaWeights);

  /**
   * COMPUTE FOR NEW WEIGHTS (LAST HIDDEN <-> OUTPUT)
   */
  tempNewWeights  = new Matrix(
                      this->topology.at(indexOutputLayer - 1),
                      this->topology.at(indexOutputLayer),
                      false
                    );

  for(int r = 0; r < this->topology.at(indexOutputLayer - 1); r++) {
    for(int c = 0; c < this->topology.at(indexOutputLayer); c++) {

      double originalValue  = this->weightMatrices.at(indexOutputLayer - 1)->getValue(r, c);
      double deltaValue     = deltaWeights->getValue(c, r);

      originalValue = this->momentum * originalValue;
      deltaValue    = this->learningRate * deltaValue;
      
      tempNewWeights->setValue(r, c, (originalValue - deltaValue));
    }
  }

  newWeights.push_back(new Matrix(*tempNewWeights));

  delete gradientsTransposed;
  delete zActivatedVals;
  delete tempNewWeights;
  delete deltaWeights;
  delete derivedValues;

  ///////////////////////////

  /**
   *  PART 2: LAST HIDDEN LAYER TO INPUT LAYER
   */
  for(int i = (indexOutputLayer - 1); i > 0; i--) {
    pGradients  = new Matrix(*gradients);
    delete gradients;

    transposedPWeights  = this->weightMatrices.at(i)->transpose();

    gradients   = new Matrix(
                    pGradients->getNumRows(),
                    transposedPWeights->getNumCols(),
                    false
                  );

    ::utils::Math::multiplyMatrix(pGradients, transposedPWeights, gradients);

    hiddenDerived       = this->layers.at(i)->matrixifyDerivedVals();

    for(int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++) {
      double  g = gradients->getValue(0, colCounter) * hiddenDerived->getValue(0, colCounter);
      gradients->setValue(0, colCounter, g);
    }

    if(i == 1) {
      zActivatedVals  = this->layers.at(0)->matrixifyVals();
    } else {
      zActivatedVals  = this->layers.at(i-1)->matrixifyActivatedVals();
    }

    transposedHidden  = zActivatedVals->transpose();

    deltaWeights      = new Matrix(
                          transposedHidden->getNumRows(),
                          gradients->getNumCols(),
                          false
                        );

    ::utils::Math::multiplyMatrix(transposedHidden, gradients, deltaWeights);

    // update weights
    tempNewWeights  = new Matrix(
                        this->weightMatrices.at(i - 1)->getNumRows(),
                        this->weightMatrices.at(i - 1)->getNumCols(),
                        false
                      );

    for(int r = 0; r < tempNewWeights->getNumRows(); r++) {
      for(int c = 0; c < tempNewWeights->getNumCols(); c++) {
        double originalValue  = this->weightMatrices.at(i - 1)->getValue(r, c);
        double deltaValue     = deltaWeights->getValue(r, c);

        originalValue = this->momentum * originalValue;
        deltaValue    = this->learningRate * deltaValue;
        
        tempNewWeights->setValue(r, c, (originalValue - deltaValue));
      }
    }

    newWeights.push_back(new Matrix(*tempNewWeights));

    delete pGradients;
    delete transposedPWeights;
    delete hiddenDerived;
    delete zActivatedVals;
    delete transposedHidden;
    delete tempNewWeights;
    delete deltaWeights;
  }
  delete gradients;

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    delete this->weightMatrices[i];
  }

  this->weightMatrices.clear();

  reverse(newWeights.begin(), newWeights.end());

  for(int i = 0; i < newWeights.size(); i++) {
    this->weightMatrices.push_back(new Matrix(*newWeights[i]));
    delete newWeights[i];
  }
}
