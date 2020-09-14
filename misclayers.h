#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

class inputlayer{
private:

public:
  vector<float> a;  //activations

  //construction code:
  inputlayer(const float& ls){
    this->a.resize(ls); //resize activations
  }
};
