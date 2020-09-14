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

  //vars for backprop:
  vector<vector<float>> tpw; //for storing transposed weights
  vector<vector<float>> nw; //nabla_w (dc/dw)
  vector<float> nb; //nabla_b (dc/db)
  vector<float> d; //delta (error)

  //construction code:
  baselayer(const float& pls, //previous layer size
            const float& ls //layer size
            ){
    //resize weights:
    this->w.resize(ls);
    for (int neuron = 0; neuron < ls; neuron++){
      this->w[neuron].resize(pls);
    }
    //resize nabla_w: (dc/dw)
    this->nw.resize(ls);
    for (int neuron = 0; neuron < ls; neuron++){
      this->nw[neuron].resize(pls);
    }
    this->b.resize(ls); //resize biases
    this->a.resize(ls); //resize activations
    this->pfa.resize(ls); //resize pre (activation) function activations
    this->d.resize(ls); //resize deltas
    this->nb.resize(ls); //resize nabla_b (dc/db)
  }


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
  }

  //feedforwards with linear (/no) activation function
  void feedforwards(const vector<float>& inp){
    for (int neuron = 0; neuron < this->a.size(); neuron++){
      dot(inp, this->w[neuron], this->pfa[neuron]);
      this->pfa[neuron] += this->b[neuron];
    }
    this->a = this->pfa;
  }

  //backprop for linear neurons:
  void backprop(const vector<float>& nd, //Next layer Delta
                const vector<vector<float>>& ntpw //Next layer TransPosed Weights
                ){
    //set own tpw so previous layer can backprop
    transpose(this->w, this->tpw);

    //calculate delta
    for (int neuron = 0; neuron < this->d.size(); neuron++){ //get dotprod of tpw and delta[layer+1]
        dot(ntpw[neuron], nd, this->d[neuron]);
      }
  }

  //update network parameters based on nw and nb
  void updateparams(const float& eta){
    //update biases
    for (int n = 0; n < this->a.size(); n++){
      this->b[n] -= (this->nb[n] * eta);
    }

    //update Weights
    for (int n1 = 0; n1 < this->w.size(); n1++){
      for (int n2 = 0; n2 < this->w[n1].size(); n2++){
        this->w[n1][n2] -= (this->nw[n1][n2] * eta);
      }
    }
  }
};
class baseoutputlayer : public baselayer{
private:

public:
  baseoutputlayer(const float& pls, //previous layer size
                  const float& ls //layer size
                  ) : baselayer(pls, ls){
      // Add any layer-specific construction code here, if necessary
  }
  //backprop for linear output neurons using MSE:
  void backprop(const vector<float>& desiredoutput){
    //set own tpw so previous layer can backprop
    transpose(this->w, this->tpw);
    //calculate delta
    MSEderivative(this->a, desiredoutput, this->d);
  }
};

class sigmoidlayer : public baselayer{
private:

public:
  sigmoidlayer(const float& pls, //previous layer size
               const float& ls //layer size
               ) : baselayer(pls, ls){
      // Add any layer-specific construction code here, if necessary
  }

  //feedforwards with sigmoid activation function
  void feedforwards(const vector<float>& inp){
    for (int neuron = 0; neuron < this->a.size(); neuron++){
      dot(inp, this->w[neuron], this->pfa[neuron]);
      this->pfa[neuron] += this->b[neuron];
    }
    vectsigmoid(this->pfa, this->a);
  }
};

int main(){
  baselayer l1(2, 2);
  baseoutputlayer l2(2, 2);

  l1.paraminit(-1, 1);
  l2.paraminit(-1, 1);

  vector<float> inp = {0.2, 0.3};
  vector<float> desiredout = {0, 0};

  for (int i = 0; i < 20; i++){
    //run a feedforwards and backprop pass:
    l1.feedforwards(inp);
    l2.feedforwards(l1.a);
    l2.backprop(desiredout);
    l1.backprop(l2.d, l2.tpw);
    //calc nabla_bs:
    l2.nb = l2.d;
    l1.nb = l1.d;
    //calc nabla_ws:
    for (int n = 0; n < l2.a.size(); n++){
      vectbyscalarmultiply(l2.a, l2.d[n], l2.nw[n]);
    }
    for (int n = 0; n < l1.a.size(); n++){
      vectbyscalarmultiply(l1.a, l1.d[n], l1.nw[n]);
    }
    //update parameters of all layers
    l1.updateparams(0.1);
    l2.updateparams(0.1);
    //print out activations to terminal:
    cout << "l2 activations:" << endl;
    printV(l2.a);
  }
  return 0;
}
