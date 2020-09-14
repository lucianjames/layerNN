#include <vector>
#include <iostream>
#include <cmath>



//=============base funcs:=============

//sigmoid function
float sigmoid(const float& a,
	     float& tomod){ //tomod will be modified by this function
  tomod = 1/(1+exp(-a));
  return tomod;
}
//sigmoid prime
float sigmoidprime(const float& a,
		  float& tomod){ //tomod will be modified by this function
  sigmoid(a, tomod);
  tomod = tomod*(1-tomod);
  return tomod;
}

//============vector funcs:============

//sigmoid function for vectors
std::vector<float> vectsigmoid(const std::vector<float>& a,
			       std::vector<float>& vecttomod){ //vecttomod will be modified by this function
  if (a.size() != vecttomod.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "vecttomod.size(): " << vecttomod.size() << std::endl;
    throw std::runtime_error("vectsigmoids received inputs were of different size");
  }
  for (int i = 0; i < a.size(); i++){
    sigmoid(a[i], vecttomod[i]);
  }
  return vecttomod;
}
//sigmoid prime for vectors
std::vector<float> vectsigmoidprime(const std::vector<float>& a,
				    std::vector<float>& vecttomod){ //vecttomod will be modified by this function
  if (a.size() != vecttomod.size()){
    std::cout << "a.size(): " << a.size() << std::endl;
    std::cout << "vecttomod.size(): " << vecttomod.size() << std::endl;
    throw std::runtime_error("vectsigmoidprimes received inputs were of different size");
  }
  for (int i = 0; i < a.size(); i++){
    sigmoidprime(a[i], vecttomod[i]);
  }
  return vecttomod;
}
