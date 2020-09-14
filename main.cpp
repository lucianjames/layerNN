#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "vectio.h"
#include "math.h"
#include "mnist.h"
#include "activfuncs.h"
#include "lossfuncs.h"
using namespace std;

class baselayer{ //base layer class, uses fully linear activation function
private:

public:
  //stuff for RNG for param init:
  random_device rd{};
  mt19937_64 generator{rd()};
  uniform_real_distribution<float> dist;

  //basic net vars:
  vector<vector<float>> w; //weights
  vector<float> b; //biases
  vector<float> a;  //activations
  vector<float> pfa; //pre (activation) function activations

  //vars for backprop + other:

  //init net params in uniform dist of specified size:
  void paraminit(const float& x,
                 const float& y){
    typename uniform_real_distribution<float>::param_type prms (x, y);
    this->dist.param (prms);

    //init weights:
    for (auto& n : this->w){
      for (auto& n2 : n){
        n2 = dist(generator);
      }
    }

    //init biases:
    for (auto& n : this->b){
      n = dist(generator);
    }

  };

  //feedforwards with linear (/no) activation function
  void feedforwards(const vector<float>& inp){
    for (int neuron = 0; neuron < this->a.size(); neuron++){
      dot(inp, this->w[neuron], this->a[neuron]);
      this->pfa[neuron] += this->b[neuron];
    }
    this->a = this->pfa;
  }

  //construction code:
  baselayer(const float& pls, //previous layer size
            const float& ls //layer size
            ){

    //resize weights:
    this->w.resize(ls);
    for (int neuron = 0; neuron < ls; neuron++){
      this->w[neuron].resize(pls);
    }

    this->b.resize(ls); //resize biases:
    this->a.resize(ls); //resize activations
    this->pfa.resize(ls); //resize pre (activation) function activations
  }

};


int main(){
  baselayer l1(2, 2);
  baselayer l2(2, 2);

  l1.paraminit(-1, 1);
  l2.paraminit(-1, 1);

  vector<float> inp = {0.3, 0.2};

  //propogate input through net:
  l1.feedforwards(inp);
  l2.feedforwards(l1.a);


  cout << "input:" << endl;
  printV(inp);

  cout << "l1 activations:" << endl;
  printV(l1.a);

  cout << "l2 activations:" << endl;
  printV(l2.a);

  return 0;
}
