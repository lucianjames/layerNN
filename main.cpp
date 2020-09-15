#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "vectio.h"
#include "math.h"
#include "mnist.h"
#include "activfuncs.h"
#include "lossfuncs.h"
#include "baselayer.h"
#include "sigmoidlayer.h"
using namespace std;

int main(){
  //load MNIST dataset:
  cout << "Loading MNIST data" << endl;
  vector<vector<float>> imgs = loadimages();
  vector<int> labels = loadlabels();
  vector<vector<float>> testimgs = loadtestimages();
  vector<int> testlabels = loadtestlabels();
  cout << "Data loaded" << endl;
  cout << "Normalising data" << endl;
  for (auto& i : imgs){
    for (auto& i2 : i){
      i2 = i2/255;
    }
  }
  for (auto& i : testimgs){
    for (auto& i2 : i){
      i2 = i2/255;
    }
  }
  cout << "Data normalised" << endl;


  //network hyperparameters:
  float eta = 1;
  int n_epochs = 10;
  //dataset info:
  int n_imgs = 60000;
  int n_timgs = 10000;


  //create vector of pointers to class objects to define the network:
  vector<baselayer*> lyrs;
  sigmoidlayer* l1 = new sigmoidlayer(784,32);
  sigmoidoutputlayer* outputlayer = new sigmoidoutputlayer(32,10);
  lyrs.push_back (l1);
  lyrs.push_back (outputlayer);

  //init network params:
  for (auto& l : lyrs){
    l->paraminit(-1, 1);
  }

  //MNIST simple gradient descent test:
  vector<float> inp;
  inp.resize(784);
  vector<float> desiredout;
  desiredout.resize(10);
  for (int epoch = 0; epoch < n_epochs; epoch++){
    for (int img = 0; img < n_imgs; img++){
      inp = imgs[img];
      for (auto& i : desiredout){
        i = 0;
      }
      desiredout[labels[img]] = 1;

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
      l1->calcnablas(inp);
      for (int l = 1; l < lyrs.size(); l++){
        lyrs[l]->calcnablas(lyrs[l-1]->a);
      }
      //update all layer params:
      for (auto& l : lyrs){
        l->updateparams(eta);
      }
    }
  }



  //run a little test:
  int correctclassifications = 0;
  for (int testimg = 0; testimg < n_timgs; testimg++){
    inp = imgs[testimg];
    //run forwards pass:
    l1->feedforwards(inp);
    for (int l = 1; l < lyrs.size(); l++){
      lyrs[l]->feedforwards(lyrs[l-1]->a);
    }
    if ((max_element(outputlayer->a.begin(), outputlayer->a.end()) - outputlayer->a.begin()) == labels[testimg]){
	     correctclassifications += 1;
      }
  }

  cout << correctclassifications << endl;

  return 0;
}
