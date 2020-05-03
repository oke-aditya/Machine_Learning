/*
The <torch/script.h> header encompasses all relevant includes from the LibTorch library necessary to run the example. 

Our application accepts the file path to a serialized PyTorch ScriptModule as its 
only command line argument and then proceeds to deserialize the module using the torch::jit::load() function, 
which takes this file path as input. 

In return we receive a torch::jit::script::Module object.
*/
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
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

  std::cout << "ok\n";
}