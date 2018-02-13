#ifndef _MISC_HPP_
#define _MISC_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>

using namespace std;

namespace utils
{
  class Misc
  {
  public:
    static vector< vector<double> > fetchData(string path);
  };
}

#endif
