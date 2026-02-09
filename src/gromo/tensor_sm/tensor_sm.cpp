#include <torch/extension.h>

// s has size N x C_{l-1} x k x k x C_{l-1} x k x k
void add_BTB_to_tensor_s_conv(torch::Tensor b, torch::Tensor &s, int kernel_size=3, int padding=0);

// m has size N x C_{l-1} x k x k x C_l
void add_BTgradA_to_tensor_m_conv(torch::Tensor b, torch::Tensor gradA, torch::Tensor &m,
    int kernel_size=3, int padding=0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_BTB_to_tensor_s_conv", &add_BTB_to_tensor_s_conv, 
        "Add covariance unfold(B)^T unfold(B) to S");

    m.def("add_BTgradA_to_tensor_m_conv", &add_BTgradA_to_tensor_m_conv,
        "Add unfold(B)^T gradA to M");
}