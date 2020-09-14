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
  virtual void paraminit(const float& x,
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
  virtual void feedforwards(const vector<float>& inp){
    for (int neuron = 0; neuron < this->a.size(); neuron++){
      dot(inp, this->w[neuron], this->pfa[neuron]);
      this->pfa[neuron] += this->b[neuron];
    }
    this->a = this->pfa;
  }

  //backprop for linear neurons:
  virtual void backprop(const vector<float>& nd, //Next layer Delta
                const vector<vector<float>>& ntpw //Next layer TransPosed Weights
                ){
    //set own tpw so previous layer can backprop
    transpose(this->w, this->tpw);

    //calculate delta
    for (int neuron = 0; neuron < this->d.size(); neuron++){ //get dotprod of tpw and delta[layer+1]
        dot(ntpw[neuron], nd, this->d[neuron]);
      }
  }

  virtual void calcnablas(){
    //calc nabla b:
    this->nb = this->d;
    //calc nabla w:
    for (int n = 0; n < this->a.size(); n++){
        vectbyscalarmultiply(this->a, this->d[n], this->nw[n]);
      }
  }

  //update network parameters based on nw and nb
  virtual void updateparams(const float& eta){
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

int main(){
  //network hyperparameters:
  float eta = 1;

  //create vector of pointers to class objects to define the network:
  vector<baselayer*> lyrs;
  sigmoidlayer* l1 = new sigmoidlayer(2,2);
  sigmoidlayer* l2 = new sigmoidlayer(2,2);
  sigmoidlayer* l3 = new sigmoidlayer(2,2);
  sigmoidlayer* l4 = new sigmoidlayer(2,2);
  sigmoidoutputlayer* outputlayer = new sigmoidoutputlayer(2,2);
  lyrs.push_back (l1);
  lyrs.push_back (l2);
  lyrs.push_back (l3);
  lyrs.push_back (l4);
  lyrs.push_back (outputlayer);

  //init network params:
  for (auto& l : lyrs){
    l->paraminit(-1, 1);
  }

  //dummy gradient descent test:
  vector<float> inp = {0.8, 0.9};
  vector<float> desiredout = {0, 0};
  for (int loops = 0; loops < 100; loops++){
    //run forwards pass:
    l1->feedforwards(inp);
    for (int l = 1; l < lyrs.size(); l++){
      lyrs[l]->feedforwards(lyrs[l-1]->a);
    }
    //run backprop pass:
    outputlayer->backprop(desiredout);
    for (int l = lyrs.size()-2; l > -1; l--){
        lyrs[l]->backprop(lyrs[l+1]->d, lyrs[l+1]->tpw);
      }
    //calc nablas for all layers:
    for (auto& l : lyrs){
      l->calcnablas();
    }
    //update all layer params:
    for (auto& l : lyrs){
      l->updateparams(eta);
    }
    //print out network activation values:
    printV(outputlayer->a);
  }
  return 0;
}
