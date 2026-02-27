#include <torch/extension.h>

// s has size N x C_{l-1} x k x k x C_{l-1} x k x k
// Computes the upper triangular part of s only
// The fastest!
void add_BTB_to_tensor_s_conv(torch::Tensor b, torch::Tensor &s, int kernel_size=3, int padding=0);

// s has size N x C_{l-1} x k x k x C_{l-1} x k x k
// Computes the full s, instead of its upper-triangular part only
// The fastest!
void add_BTB_to_tensor_s_conv_full(torch::Tensor b, torch::Tensor &s, int kernel_size=3, int padding=0);

// m has size N x C_{l-1} x k x k x C_l
void add_BTgradA_to_tensor_m_conv(torch::Tensor b, torch::Tensor gradA, torch::Tensor &m,
    int kernel_size=3, int padding=0);

// s has size C_{l-1} x k x k x C_{l-1} x k x k
void add_BTB_to_tensor_s_conv_v2(torch::Tensor b, torch::Tensor &s, int kernel_size=3, int padding=0);

// s has size C_{l-1} x k x k x C_{l-1} x k x k
void add_BTB_to_tensor_s_conv_v2_dim3(torch::Tensor b, torch::Tensor &s, int kernel_size=3, int padding=0);

void add_BTB_to_aux(torch::Tensor b, torch::Tensor &aux, int kernel_size=3, int padding=0);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_BTB_to_tensor_s_conv", &add_BTB_to_tensor_s_conv, 
        "Add covariance unfold(B)^T unfold(B) to S (upper-triangular part)");

    m.def("add_BTB_to_tensor_s_conv_full", &add_BTB_to_tensor_s_conv_full, 
        "Add covariance unfold(B)^T unfold(B) to S (full)");

    m.def("add_BTgradA_to_tensor_m_conv", &add_BTgradA_to_tensor_m_conv,
        "Add unfold(B)^T gradA to M");

    m.def("add_BTB_to_tensor_s_conv_v2", &add_BTB_to_tensor_s_conv_v2, 
        "Add covariance unfold(B)^T unfold(B) to S");

    m.def("add_BTB_to_tensor_s_conv_v2_dim3", &add_BTB_to_tensor_s_conv_v2_dim3, 
        "Add covariance unfold(B)^T unfold(B) to S");

    m.def("add_BTB_to_aux", &add_BTB_to_aux, 
        "Add covariance unfold(B)^T unfold(B) to aux");
}