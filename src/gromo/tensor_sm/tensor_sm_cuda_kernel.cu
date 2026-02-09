#include <torch/extension.h>

#define THREADS_PER_BLOCK 1024

// Rounded up division, to compute number of thread blocks when launching CUDA kernels
int divUp(int a, int b)
{
    return (a + b - 1)/b;
}

template <typename T> __global__ void add_BTB_to_tensor_s_conv_kernel(
    int batch_size, int C, int height, int width, int kernel_size, int padding,
    const T *b, T *s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = C*kernel_size*kernel_size;
    int uptri_size = N*(N+1)/2;
    if (idx>=batch_size*uptri_size)
        return;
    
    int n = idx/uptri_size;
    int remainder = idx%uptri_size;
    int uptri_row=0, uptri_col=0;
    int length_row=N;
    while (remainder>=length_row)
    {
        uptri_row++;
        remainder-=length_row;
        length_row--;
    }
    uptri_col = uptri_row+remainder;
    int c1 = uptri_row/(kernel_size*kernel_size);
    int ki1 = (uptri_row%(kernel_size*kernel_size))/kernel_size;
    int kj1 = uptri_row%kernel_size;
    
    int c2 = uptri_col/(kernel_size*kernel_size);
    int ki2 = (uptri_col%(kernel_size*kernel_size))/kernel_size;
    int kj2 = uptri_col%kernel_size;
    
    int height_out = height-kernel_size+1, width_out = width-kernel_size+1;
 
    int offset_c1 = c1*height*width, offset_c2 = c2*height*width; 
    int offsetN = C*height*width;
    
    T sum = 0;
    const T *b1 = b+n*offsetN+offset_c1;
    const T *b2 = b+n*offsetN+offset_c2;
    for (int i=0;i<height_out;i++)
        for (int j=0;j<width_out;j++)
            sum += b1[(ki1+i)*width + kj1+j]*b2[(ki2+i)*width + kj2+j];

    s[n*N*N+uptri_row*N+uptri_col] += sum;
}

void add_BTB_to_tensor_s_conv(torch::Tensor b, torch::Tensor &s, int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && s.is_cuda() && s.is_contiguous());
    assert(b.dtype()==s.dtype());

    // Extra assertions needed
    
    int batch_size = b.size(0), C = b.size(1), height = b.size(2), width = b.size(3);
    
    int N=C*kernel_size*kernel_size;
    int uptri_size = N*(N+1)/2;

    int blocksPerGrid = divUp(batch_size*uptri_size, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_kernel<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<float>(), s.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_kernel<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<double>(), s.data_ptr<double>());
    }
}

template <typename T> __global__ void add_BTgradA_to_tensor_m_conv_kernel(
    int batch_size, int Cin, int Cout, int height, int width, int kernel_size, int padding,
    const T *b, const T *gradA, T *m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx>=batch_size*Cin*kernel_size*kernel_size*Cout)
        return;
    
    int n, c_in, ki, kj, c_out, remainder;
    n = idx/(Cin*kernel_size*kernel_size*Cout);
    remainder = idx%(Cin*kernel_size*kernel_size*Cout);
    
    c_in = remainder/(kernel_size*kernel_size*Cout);
    remainder %= (kernel_size*kernel_size*Cout);
    
    ki = remainder/(kernel_size*Cout);
    remainder %= (kernel_size*Cout);
    
    kj = remainder/Cout;
    c_out = remainder%Cout;

    int height_out = height-kernel_size+1, width_out = width-kernel_size+1;
    int height_gradA = height_out+2*padding, width_gradA = width_out+2*padding;
    int gradA_start = padding;

    T sum = 0;
    const T *b_offset = b + n*Cin*height*width + c_in*height*width;
    const T *gradA_offset = gradA + n*Cout*height_gradA*width_gradA + c_out*height_gradA*width_gradA;
    for (int i=0;i<height_out;i++)
        for (int j=0;j<width_out;j++)
            sum -= b_offset[(ki+i)*width + kj+j]
                * gradA_offset[(i+gradA_start)*width_gradA + j+gradA_start];
    m[idx] += sum;
}

void add_BTgradA_to_tensor_m_conv(torch::Tensor b, torch::Tensor gradA, torch::Tensor &m,
    int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && gradA.is_cuda() && gradA.is_contiguous()
        && m.is_cuda() && m.is_contiguous());
    assert(b.dtype()==m.dtype() && b.dtype()==gradA.dtype());

    // Extra assertions needed

    int batch_size = b.size(0), Cin = b.size(1), Cout=gradA.size(1),
        height = b.size(2), width = b.size(3);
    
    int blocksPerGrid = divUp(batch_size*Cin*kernel_size*kernel_size*Cout, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTgradA_to_tensor_m_conv_kernel<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, Cin, Cout, height, width, kernel_size, padding,
            b.data_ptr<float>(), gradA.data_ptr<float>(), m.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTgradA_to_tensor_m_conv_kernel<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, Cin, Cout, height, width, kernel_size, padding,
            b.data_ptr<double>(), gradA.data_ptr<double>(), m.data_ptr<double>());
    }
}