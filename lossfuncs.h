#include <vector>
#include <cmath>


//=======all-round error funcs:========

float MSE(const vector<float>& outactivs,
	  const vector<float>& desiredout,
	  float& cost){
  cost = 0;
  for (int neuron = 0; neuron < outactivs.size(); neuron++){
    cost += pow((outactivs[neuron]-desiredout[neuron]) ,2);
  }
  return cost;
}

float CE(const vector<float>& outactivs,
	 const vector<float>& desiredout,
	 float& cost){
  cost = 0;
  for (int neuron = 0; neuron < outactivs.size(); neuron++){
    cost += (desiredout[neuron] * log(outactivs[neuron])) + (1-desiredout[neuron]*log(1-outactivs[neuron])); 
  }
  return cost;
}




//======element-wise derivative funcs:=======

std::vector<float> CEderivative(const std::vector<float>& outputactivations,
					  const std::vector<float>& desiredoutput,
					  std::vector<float>& nabla_c){

  for (int i = 0; i < outputactivations.size(); i++){
    nabla_c[i] = -(desiredoutput[i]/outputactivations[i]) + ((1-desiredoutput[i])/(1-outputactivations[i]));
  }
  
  return nabla_c;
}

std::vector<float> MSEderivative(const std::vector<float>& outputactivations,
				 const std::vector<float>& desiredoutput,
				 std::vector<float>& nabla_c){
  
  vectsub(outputactivations, desiredoutput, nabla_c);

  return nabla_c;
}
