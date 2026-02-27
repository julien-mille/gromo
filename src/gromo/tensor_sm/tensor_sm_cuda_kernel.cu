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


template <typename T> __global__ void add_BTB_to_tensor_s_conv_full_kernel(
    int batch_size, int C, int height, int width, int kernel_size, int padding,
    const T *b, T *s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int Ck2 = C*kernel_size*kernel_size;
    if (idx>=batch_size*Ck2*Ck2)
        return;
    
    int n = idx/(Ck2*Ck2);
    int remainder = idx%(Ck2*Ck2);
    
    int c1 = remainder/(kernel_size*kernel_size*Ck2);
    remainder %= kernel_size*kernel_size*Ck2;
    int ki1 = remainder/(kernel_size*Ck2);
    remainder %= kernel_size*Ck2;
    int kj1 = remainder/Ck2;
    remainder %= Ck2;
    
    int c2 = remainder/(kernel_size*kernel_size);
    remainder %= kernel_size*kernel_size;
    int ki2 = remainder/kernel_size;
    int kj2 = remainder%kernel_size;
    
    int height_out = height-kernel_size+1, width_out = width-kernel_size+1;
 
    int offset_c1 = c1*height*width, offset_c2 = c2*height*width; 
    int offsetN = C*height*width;
    
    T sum = 0;
    const T *b1 = b+n*offsetN+offset_c1;
    const T *b2 = b+n*offsetN+offset_c2;
    for (int i=0;i<height_out;i++)
        for (int j=0;j<width_out;j++)
            sum += b1[(ki1+i)*width + kj1+j]*b2[(ki2+i)*width + kj2+j];

    s[idx] += sum;
}

void add_BTB_to_tensor_s_conv_full(torch::Tensor b, torch::Tensor &s, int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && s.is_cuda() && s.is_contiguous());
    assert(b.dtype()==s.dtype());

    // Extra assertions needed
    
    int batch_size = b.size(0), C = b.size(1), height = b.size(2), width = b.size(3);
    int Ck2=C*kernel_size*kernel_size;
    
    int blocksPerGrid = divUp(batch_size*Ck2*Ck2, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_full_kernel<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<float>(), s.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_full_kernel<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
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

template <typename T> __global__ void add_BTB_to_tensor_s_conv_v2_kernel(
    int batch_size, int C, int height, int width, int kernel_size, int padding,
    const T *b, T *aux)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int uptri_size = C*(C+1)/2;
    
    int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    if (idx>=batch_size*uptri_size*height_out*width_out*num_offsets)
        return;
    
    int n = idx/uptri_size;
    
    n = idx/(uptri_size*height_out*width_out*num_offsets);
    int remainder = idx%(uptri_size*height_out*width_out*num_offsets);
    
    int uptri_c = remainder/(height_out*width_out*num_offsets);
    
    int c1=0, c2=0;
    int length_row=C;
    while (uptri_c>=length_row)
    {
        c1++;
        uptri_c-=length_row;
        length_row--;
    }
    c2 = c1+uptri_c;
    
    remainder %= height_out*width_out*num_offsets;

    int i = remainder/(width_out*num_offsets) + kernel_size-1;
    remainder %= width_out*num_offsets;
    int j = remainder/num_offsets + kernel_size-1;
    remainder %= num_offsets;

    int offsets_i[] = {0,0,0,1,1,1,1,1,2,2,2,2,2};
    int offsets_j[] = {0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};

    int ki = offsets_i[remainder];
    int kj = offsets_j[remainder];
    
    const T *b_offset_n = b+n*C*height*width;
    aux[idx] = b_offset_n[c1*height*width + i*width + j] * b_offset_n[c2*height*width + (i+ki)*width + j+kj];
}

// aux has size up_tri x num_offsets
// s has size C x ks x ks x C x ks x ks
template <typename T> __global__ void add_BTB_to_tensor_s_conv_v2_kernel_aux_sum_to_s(
    int C, int kernel_size, const T *aux_sum, T *s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int uptri_size = C*(C+1)/2;
    
    // int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    if (idx>=C*kernel_size*kernel_size*C*kernel_size*kernel_size)
        return;
    
    int c1 = idx/(kernel_size*kernel_size*C*kernel_size*kernel_size);
    int remainder = idx%(kernel_size*kernel_size*C*kernel_size*kernel_size);

    int ki1 = remainder/(kernel_size*C*kernel_size*kernel_size);
    remainder %= kernel_size*C*kernel_size*kernel_size;

    int kj1 = remainder/(C*kernel_size*kernel_size);
    remainder %= C*kernel_size*kernel_size;
    
    int c2 = remainder/(kernel_size*kernel_size);
    remainder %= kernel_size*kernel_size;
    
    int ki2 = remainder/kernel_size;
    int kj2 = remainder%kernel_size;
    
    if (c2<c1)
    {
        int ctmp = c2;
        c2 = c1;
        c1 = ctmp;
    }
    int cc=0;
    for (int i=0;i<c1;i++)
        cc+=C-i;
    cc+=c2-c1;

    /*
    n = idx/(uptri_size*height_out*width_out*num_offsets);
    remainder = idx%(uptri_size*height_out*width_out*num_offsets);
    
    int uptri_c = remainder/(height_out*width_out*num_offsets);
    height_out*width_out*num_offsets

    int c1=0, c2=0;
    int length_row=C;
    while (uptri_c>=length_row)
    {
        c1++;
        uptri_c-=length_row;
        length_row--;
    }
    c2 = c1+uptri_c;
    
    remainder %= height_out*width_out*num_offsets;

    int i = remainder/(width_out*num_offsets) + kernel_size-1;
    remainder %= wiidth_out*num_offsets;
    int j = remainder/num_offsets + kernel_size-1;
    remainder %= num_offsets;

    int offsets_i[] = [0,0,0,1,1,1,1,1,2,2,2,2,2];
    int offsets_j[] = [0,1,2,-2,-1,0,1,2,-2,-1,0,1,2];

    int ki = offsets_i[remainder];
    int kj = offsets_j[remainder];
    */
    
    // aux_sum[idx] = x_offset_n[c1*height*width + i*width + j] * x_offset_n[c2*height*width + (i+ki)*width + j+kj];
    int offset_i, offset_j, final_offset;
    if (ki2<ki1)
    {
        offset_i = ki1-ki2;
        offset_j = kj1-kj2;
    }
    else {
        offset_i = ki2-ki1;
        offset_j = kj2-kj1;
    }
    if (offset_i==0)
        final_offset = offset_j;
    else {
        final_offset = 3+(offset_i-1)*5+offset_j+2;
    }

    s[idx] += aux_sum[cc*num_offsets + final_offset];
}

void add_BTB_to_tensor_s_conv_v2(torch::Tensor b, torch::Tensor &s, int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && s.is_cuda() && s.is_contiguous());
    assert(b.dtype()==s.dtype());

    // Extra assertions needed
    
    int batch_size = b.size(0), C = b.size(1), height = b.size(2), width = b.size(3);
    
    int uptri_size = C*(C+1)/2;

    int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    torch::Tensor aux = torch::empty({batch_size,uptri_size,height_out,width_out,num_offsets}, b.options());

    int blocksPerGrid = divUp(batch_size*uptri_size*height_out*width_out*num_offsets, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_v2_kernel<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<float>(), aux.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_v2_kernel<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<double>(), aux.data_ptr<double>());
    }

    torch::Tensor aux_sum = aux.sum({0,2,3});
    assert(aux_sum.is_contiguous());

    blocksPerGrid = divUp(C*kernel_size*kernel_size*C*kernel_size*kernel_size, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_v2_kernel_aux_sum_to_s<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            C, kernel_size, aux_sum.data_ptr<float>(), s.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_v2_kernel_aux_sum_to_s<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            C, kernel_size, aux_sum.data_ptr<double>(), s.data_ptr<double>());
    }
}


// With dim3
template <typename T> __global__ void add_BTB_to_tensor_s_conv_v2_dim3_kernel(
    int batch_size, int C, int height, int width, int kernel_size, int padding,
    const T *b, T *aux)
{
    /*
    int blocksX = divUp(batch_size, 16);
    int blocksY = divUp(uptri_size*height_out, 8);
    int blocksZ = divUp(width_out*num_offsets, 8);
    */
    
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n>=batch_size)
        return;

    int uptri_size = C*(C+1)/2;
    
    int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    int idx1 = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx1>=uptri_size*height_out)
        return;
    
    int idx2 = blockDim.z * blockIdx.z + threadIdx.z;

    if (idx2>=width_out*num_offsets)
        return;

    // int remainder = idx%(uptri_size*height_out*width_out*num_offsets);
    int uptri_c = idx1/height_out;
    
    int c1=0, c2=0;
    int length_row=C;
    while (uptri_c>=length_row)
    {
        c1++;
        uptri_c-=length_row;
        length_row--;
    }
    c2 = c1+uptri_c;
    
    int i = idx1-uptri_c*height_out + kernel_size-1;
    int j = idx2/num_offsets + kernel_size-1;

    int remainder = idx2-j*num_offsets;

    int offsets_i[] = {0,0,0,1,1,1,1,1,2,2,2,2,2};
    int offsets_j[] = {0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};

    int ki = offsets_i[remainder];
    int kj = offsets_j[remainder];
    
    const T *b_offset_n = b+n*C*height*width;
    aux[n*uptri_size*height_out*width_out*num_offsets + idx1*width_out*num_offsets + idx2]
        = b_offset_n[c1*height*width + i*width + j] * b_offset_n[c2*height*width + (i+ki)*width + j+kj];
}

// aux has size up_tri x num_offsets
// s has size C x ks x ks x C x ks x ks
template <typename T> __global__ void add_BTB_to_tensor_s_conv_v2_dim3_kernel_aux_sum_to_s(
    int C, int kernel_size, const T *aux_sum, T *s)
{
    int idx1 = blockDim.x * blockIdx.x + threadIdx.x;
    int idx2 = blockDim.y * blockIdx.y + threadIdx.y;

    // int uptri_size = C*(C+1)/2;
    
    // int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    if (idx1>=C*kernel_size*kernel_size || idx2>=C*kernel_size*kernel_size)
        return;
    
    // No modulo
    int c1 = idx1/(kernel_size*kernel_size);
    int remainder = idx1-c1*kernel_size*kernel_size;
    
    int ki1 = remainder/kernel_size;
    int kj1 = remainder-ki1*kernel_size;

    int c2 = idx2/(kernel_size*kernel_size);
    remainder = idx2-c2*kernel_size*kernel_size;
    
    int ki2 = remainder/kernel_size;
    int kj2 = remainder-ki2*kernel_size;
    
    if (c2<c1)
    {
        int ctmp = c2;
        c2 = c1;
        c1 = ctmp;
    }
    int cc=0;
    for (int i=0;i<c1;i++)
        cc+=C-i;
    cc+=c2-c1;

    // aux_sum[idx] = x_offset_n[c1*height*width + i*width + j] * x_offset_n[c2*height*width + (i+ki)*width + j+kj];
    int offset_i, offset_j, final_offset;
    if (ki2<ki1)
    {
        offset_i = ki1-ki2;
        offset_j = kj1-kj2;
    }
    else {
        offset_i = ki2-ki1;
        offset_j = kj2-kj1;
    }
    if (offset_i==0)
        final_offset = offset_j;
    else {
        final_offset = 3+(offset_i-1)*5+offset_j+2;
    }

    s[idx1*C*kernel_size*kernel_size + idx2] += aux_sum[cc*num_offsets + final_offset];
}

void add_BTB_to_tensor_s_conv_v2_dim3(torch::Tensor b, torch::Tensor &s, int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && s.is_cuda() && s.is_contiguous());
    assert(b.dtype()==s.dtype());

    // Extra assertions needed
    
    int batch_size = b.size(0), C = b.size(1), height = b.size(2), width = b.size(3);
    
    int uptri_size = C*(C+1)/2;

    int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    torch::Tensor aux = torch::empty({batch_size,uptri_size,height_out,width_out,num_offsets}, b.options());

    int blocksX = divUp(batch_size, 16);
    int blocksY = divUp(uptri_size*height_out, 8);
    int blocksZ = divUp(width_out*num_offsets, 8);

    dim3 blocksPerGrid(blocksX, blocksY, blocksZ), threadsPerBlock(16,8,8);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_v2_dim3_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<float>(), aux.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_v2_dim3_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<double>(), aux.data_ptr<double>());
    }

    torch::Tensor aux_sum = aux.sum({0,2,3});
    assert(aux_sum.is_contiguous());

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    threadsPerBlock.z = 1;
    
    blocksPerGrid.x = divUp(C*kernel_size*kernel_size, threadsPerBlock.x);
    blocksPerGrid.y = divUp(C*kernel_size*kernel_size, threadsPerBlock.y);
    blocksPerGrid.z = 1;

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_tensor_s_conv_v2_dim3_kernel_aux_sum_to_s<float><<<blocksPerGrid, threadsPerBlock>>>(
            C, kernel_size, aux_sum.data_ptr<float>(), s.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_tensor_s_conv_v2_dim3_kernel_aux_sum_to_s<double><<<blocksPerGrid, threadsPerBlock>>>(
            C, kernel_size, aux_sum.data_ptr<double>(), s.data_ptr<double>());
    }
}

// Providing aux directly
template <typename T> __global__ void add_BTB_to_aux_kernel(
    int batch_size, int C, int height, int width, int kernel_size, int padding,
    const T *b, T *aux)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int uptri_size = C*(C+1)/2;
    
    int height_out=height-2*(kernel_size-1), width_out=width-2*(kernel_size-1);
    int ks_extended = 2*kernel_size-1;
    int num_offsets = (ks_extended*ks_extended)/2+1;

    if (idx>=batch_size*uptri_size*height_out*width_out*num_offsets)
        return;
    
    int n = idx/uptri_size;
    
    n = idx/(uptri_size*height_out*width_out*num_offsets);
    int remainder = idx%(uptri_size*height_out*width_out*num_offsets);
    
    int uptri_c = remainder/(height_out*width_out*num_offsets);
    
    int c1=0, c2=0;
    int length_row=C;
    while (uptri_c>=length_row)
    {
        c1++;
        uptri_c-=length_row;
        length_row--;
    }
    c2 = c1+uptri_c;
    
    remainder %= height_out*width_out*num_offsets;

    int i = remainder/(width_out*num_offsets) + kernel_size-1;
    remainder %= width_out*num_offsets;
    int j = remainder/num_offsets + kernel_size-1;
    remainder %= num_offsets;

    int offsets_i[] = {0,0,0,1,1,1,1,1,2,2,2,2,2};
    int offsets_j[] = {0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};

    int ki = offsets_i[remainder];
    int kj = offsets_j[remainder];
    
    const T *b_offset_n = b+n*C*height*width;
    aux[idx] += b_offset_n[c1*height*width + i*width + j] * b_offset_n[c2*height*width + (i+ki)*width + j+kj];
}

void add_BTB_to_aux(torch::Tensor b, torch::Tensor &aux, int kernel_size, int padding)
{
    assert(b.is_cuda() && b.is_contiguous() && aux.is_cuda() && aux.is_contiguous());
    assert(b.dtype()==aux.dtype());

    // Extra assertions needed
    
    int batch_size = b.size(0), C = b.size(1), height = b.size(2), width = b.size(3);
    
    int uptri_size = aux.size(1), height_out=aux.size(2), width_out=aux.size(3), num_offsets=aux.size(4);

    assert (height_out==height-2*(kernel_size-1) && width_out==width-2*(kernel_size-1));
    int ks_extended = 2*kernel_size-1;
    assert (num_offsets == (ks_extended*ks_extended)/2+1);

    // torch::Tensor aux = torch::empty({batch_size,uptri_size,height_out,width_out,num_offsets}, b.options());

    int blocksPerGrid = divUp(batch_size*uptri_size*height_out*width_out*num_offsets, THREADS_PER_BLOCK);

    if (b.dtype()==torch::kFloat32)
    {
        add_BTB_to_aux_kernel<float><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<float>(), aux.data_ptr<float>());
    }
    else if (b.dtype()==torch::kFloat64)
    {
        add_BTB_to_aux_kernel<double><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            batch_size, C, height, width, kernel_size, padding,
            b.data_ptr<double>(), aux.data_ptr<double>());
    }
}
