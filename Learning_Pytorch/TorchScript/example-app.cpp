/*
The <torch/script.h> header encompasses all relevant includes from the LibTorch library necessary to run the example. 

Our application accepts the file path to a serialized PyTorch ScriptModule as its 
only command line argument and then proceeds to deserialize the module using the torch::jit::load() function, 
which takes this file path as input. 

In return we receive a torch::jit::script::Module object.
*/
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) 
{
	std::cout<<"Exection started \n";
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

//   Step 4: Executing the Script Module in C++
// Having successfully loaded our serialized ResNet18 in C++, 
// we are now just a couple lines of code away from executing it! Letâ€™s add those lines to our C++ applicationâ€™s main() function:

  std::cout << "ok\n";
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  // The first two lines set up the inputs to our model. 
  // We create a vector of torch::jit::IValue (a type-erased value type script::Module methods accept and return) and add a single input. 
  // To create the input tensor, we use torch::ones(), the equivalent to torch.ones in the C++ API. 
  // We then run the script::Moduleâ€™s forward method, passing it the input vector we created. 
  // In return we get a new IValue, which we convert to a tensor by calling toTensor().
//  
//  Step 5: Getting Help and Exploring the API
//
//This tutorial has hopefully equipped you with a general understanding of a PyTorch model’s path from Python to C++. With the concepts described in this tutorial, you should be able to go from a vanilla, “eager” PyTorch model, to a compiled ScriptModule in Python, to a serialized file on disk and – to close the loop – to an executable script::Module in C++.
//
//Of course, there are many concepts we did not cover. 
//For example, you may find yourself wanting to extend your ScriptModule with a custom operator implemented in C++ or CUDA, 
//and executing this custom operator inside your ScriptModule loaded in your pure C++ production environment. The good news is: 
//this is possible, and well supported! For now, you can explore this folder for examples, 
//and we will follow up with a tutorial shortly. In the time being, the following links may be generally helpful:

}
