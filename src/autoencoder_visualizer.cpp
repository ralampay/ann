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
  cout << "autoencoder_classify [weightFile]" << endl;
}

int main(int argc, char **argv) {

  if(argc != 2) {
    printSyntax();
    exit(-1);
  }

  return 0;
}
