#include "../../include/utils/Misc.hpp"

vector< vector<double> > utils::Misc::fetchData(string path) {
  vector< vector<double> > data;

  ifstream infile(path);

  string line;
  while(getline(infile, line)) {
    vector<double>  dRow;
    string          tok;
    stringstream    ss(line);

    while(getline(ss, tok, ',')) {
      dRow.push_back(stof(tok));
    }

    data.push_back(dRow);
  }

  return data;
}
