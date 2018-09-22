#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall(err)		__cudaSafeCall(err, __FILE__, __LINE__, deviceID)
#define CudaSyncAndCheckError()	__cudaSyncAndCheckError(__FILE__, __LINE__, deviceID)

__host__ inline std::string __cudaSafeCall(cudaError err, const char *file, const int line, const int deviceID)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
		return "CUDA device ID [" + std::to_string(deviceID) + "] encountered an error: " + cudaGetErrorString(err);
	else
#endif //CUDA_ERROR_CHECK
		return "";
}

__host__ inline std::string __cudaSyncAndCheckError(const char *file, const int line, const int deviceID)
{
	cudaError_t response{ cudaSuccess };
	std::string cudaErrors{ "" };

#ifdef CUDA_ERROR_CHECK
	response = cudaGetLastError();
	if (response != cudaSuccess)
	{
		while (response != cudaSuccess)
		{
			if (!cudaErrors.empty()) cudaErrors += " <- ";
			cudaErrors += cudaGetErrorString(response);
			response = cudaGetLastError();
		}
		return "CUDA device ID [" + std::to_string(deviceID) + "] encountered an error: " + cudaErrors;
	}
#endif //CUDA_ERROR_CHECK

	response = cudaDeviceSynchronize();
	if (response != cudaSuccess)
	{
		while (response != cudaSuccess)
		{
			if (!cudaErrors.empty()) cudaErrors += " <- ";
			cudaErrors += cudaGetErrorString(response);
			response = cudaGetLastError();
		}
		return "CUDA device ID [" + std::to_string(deviceID) + "] encountered an error: " + cudaErrors;
	}
	return "";
}
