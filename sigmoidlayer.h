#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

class sigmoidlayer : public baselayer{
private:

public:
  vector<float> sigp; //sigmoid prime
  sigmoidlayer(const float& pls, //previous layer size
               const float& ls //layer size
               ) : baselayer(pls, ls){
      this->sigp.resize(ls);
  }

  //feedforwards with sigmoid activation function
  void feedforwards(const vector<float>& inp){
    for (int neuron = 0; neuron < this->a.size(); neuron++){
      dot(inp, this->w[neuron], this->pfa[neuron]);
      this->pfa[neuron] += this->b[neuron];
    }
    vectsigmoid(this->pfa, this->a);
  }

  //backprop for sigmoid neurons:
  void backprop(const vector<float>& nd, //Next layer Delta
                const vector<vector<float>>& ntpw //Next layer TransPosed Weights
                ){
    //set own tpw so previous layer can backprop
    transpose(this->w, this->tpw);
    //calculate delta
    for (int neuron = 0; neuron < this->d.size(); neuron++){ //get dotprod of tpw and delta[layer+1]
        dot(ntpw[neuron], nd, this->d[neuron]);
      }
    vectsigmoidprime(this->pfa, this->sigp );
    hadamard(this->d, this->sigp , this->d);
  }
};

class sigmoidoutputlayer : public sigmoidlayer{
private:

public:
  sigmoidoutputlayer(const float& pls, //previous layer size
                     const float& ls //layer size
                    ) : sigmoidlayer(pls, ls){
      // Add any layer-specific construction code here, if necessary
  }
  //backprop for sigmoid output neurons:
  void backprop(const vector<float>& desiredoutput){
    //set own tpw so previous layer can backprop
    transpose(this->w, this->tpw);
    //calculate delta
    MSEderivative(this->a, desiredoutput, this->d);
    vectsigmoidprime(this->pfa, this->sigp );
    hadamard(this->d, this->sigp , this->d);
  }
};
