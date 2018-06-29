#pragma unmanaged
#include <iostream>
#include <cuda_runtime.h>

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__, deviceID )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__, deviceID )

__host__ inline void __cudaSafeCall(cudaError err, const char *file, const int line, int deviceID)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		std::cerr << "\nCUDA device ID: " << deviceID
			<< " encountered an error in file '" << file
			<< "' in line " << line
			<< " : " << cudaGetErrorString(err) << ".\n";
		exit(EXIT_FAILURE);
	}
#endif

	return;
}

__host__ inline void __cudaCheckError(const char *file, const int line, int deviceID)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cerr << "\nCUDA device ID: " << deviceID
			<< " encountered an error in file '" << file
			<< "' in line " << line
			<< " : " << cudaGetErrorString(err) << ".\n";
		exit(EXIT_FAILURE);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		std::cerr << "\nCUDA device ID: " << deviceID
			<< " encountered an error after sync in file '" << file
			<< "' in line " << line
			<< " : " << cudaGetErrorString(err) << ".\n";
		exit(EXIT_FAILURE);
	}
#endif

	return;
}