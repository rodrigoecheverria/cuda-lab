#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());
    thrust::fill(temp.begin(), temp.end(), A);
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}


extern "C"
__global__ void saxpy ( const float *X, float *Y, float A, int N) 
{
  int i= threadIdx.x+blockDim.x*blockIdx.x;
  int T= blockDim.x*gridDim.x;
  #pragma unroll 4
  for (; i<N; i+=T)
    Y[i] = A*X[i] + Y[i];
}

#ifndef BLK_SZ
#define BLK_SZ 512
#endif

#ifndef THR
#define THR (BLK_SZ*6)
#endif

void saxpy_cuda(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  saxpy <<< (THR-1)/BLK_SZ + 1, BLK_SZ >>> ( thrust::raw_pointer_cast(&(X[0])), 
  					     thrust::raw_pointer_cast(&(Y[0])), A, X.size());
}

int main(int argc, char **argv)
{
  // Default Size of input vectors
  int N=1000000;

  // Modify size of input vectors using program argument
  if (argc>1) {  N = atoi(argv[1]); }

  thrust::host_vector<float> x_h(N);
  thrust::host_vector<float> y_h(N);
  thrust::host_vector<float> y_result(N);

  thrust::sequence(x_h.begin(), x_h.end(), 10.0f, 1.5f);
  thrust::fill    (y_h.begin(), y_h.end(), -2.0f);

  thrust::device_vector<float> x(x_h.begin(), x_h.end());
  thrust::device_vector<float> y(y_h.begin(), y_h.end());

  thrust::transform(x_h.begin(), x_h.end(), y_h.begin(), y_h.begin(), saxpy_functor(2.0f));

  saxpy_slow(2.0f, x, y); 
  y_result = y;
  float R = thrust::inner_product( y_h.begin(), y_h.end(), y_result.begin(), 0.0f,
                                   thrust::plus<float>(), thrust::minus<float>());
  std::cout << "Saxpy Slow. Sum of Differences is " << R << std::endl;

  thrust::transform(x_h.begin(), x_h.end(), y_h.begin(), y_h.begin(), saxpy_functor(2.0f));
  saxpy_fast(2.0f, x, y);
  y_result = y;
  R = thrust::inner_product( y_h.begin(), y_h.end(), y_result.begin(), 0.0f,
                             thrust::plus<float>(), thrust::minus<float>());
  std::cout << "Saxpy Fast. Sum of Differences is " << R << std::endl;

  thrust::transform(x_h.begin(), x_h.end(), y_h.begin(), y_h.begin(), saxpy_functor(2.0f));
  saxpy_cuda(2.0f, x, y);
  y_result = y;
  R = thrust::inner_product( y_h.begin(), y_h.end(), y_result.begin(), 0.0f,
                             thrust::plus<float>(), thrust::minus<float>());
  std::cout << "Saxpy CUDA. Sum of Differences is " << R << std::endl;

  return 0;
}

