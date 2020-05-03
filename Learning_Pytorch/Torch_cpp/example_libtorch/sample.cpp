#include <torch/torch.h>
#include <iostream>

using namespace std;

int main() {

  torch::Tensor tensor = torch::randn({3,3});
  cout << "The random matrix is:" << endl << tensor << endl;


  // Initialize the device to CPU
  torch::DeviceType device = torch::kCPU;
  // If CUDA is available,run on GPU
  if (torch::cuda::is_available())
      device = torch::kCUDA;
  cout << "Running on: "
            << (device == torch::kCUDA ? "GPU" : "CPU") << endl;

  return 0;

}
