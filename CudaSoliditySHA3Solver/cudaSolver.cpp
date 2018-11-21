/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "cudaErrorCheck.cu"
#include "cudaSolver.h"

namespace CUDASolver
{
	// --------------------------------------------------------------------
	// Static
	// --------------------------------------------------------------------

	bool CudaSolver::FoundNvAPI64()
	{
		return NV_API::FoundNvAPI64();
	}

	void CudaSolver::GetDeviceCount(int *deviceCount, const char *errorMessage)
	{
		if (!CudaCheckError(cudaGetDeviceCount(deviceCount), errorMessage))
			return;

		if (*deviceCount < 1)
		{
			int runtimeVersion = 0;
			if (!CudaCheckError(cudaRuntimeGetVersion(&runtimeVersion), errorMessage))
				return;

			auto errorMsg = std::string("There are no available device(s) that support CUDA (requires: 9.2, current: "
							+ std::to_string(runtimeVersion / 1000) + "." + std::to_string((runtimeVersion % 100) / 10) + ")");

			auto errorMsgChar = errorMsg.c_str();

			std::memcpy((void *)errorMessage, errorMsgChar, errorMsg.length());
			std::memset((void *)&errorMessage[errorMsg.length()], 0, 1);
		}
	}

	void CudaSolver::GetDeviceName(int deviceID, const char *deviceName, const char *errorMessage)
	{
		cudaDeviceProp devProp;

		if (!CudaCheckError(cudaGetDeviceProperties(&devProp, deviceID), errorMessage))
			return;

		std::string devName{ devProp.name };
		std::memcpy((void *)deviceName, devProp.name, devName.length());
		std::memset((void *)&deviceName[devName.length()], 0, 1);
	}

	// --------------------------------------------------------------------
	// Public
	// --------------------------------------------------------------------

	CudaSolver::CudaSolver() noexcept
	{
		if (NV_API::FoundNvAPI64())
			NV_API::initialize();
	}

	CudaSolver::~CudaSolver() noexcept
	{
		NV_API::unload();
	}

	void CudaSolver::GetDeviceProperties(DeviceCUDA *device, const char *errorMessage)
	{
		struct cudaDeviceProp deviceProp;
		if (!CudaCheckError(cudaGetDeviceProperties(&deviceProp, device->DeviceID), errorMessage))
			return;

		char pciBusID_s[13];
		if (!CudaCheckError(cudaDeviceGetPCIBusId(pciBusID_s, 13, device->DeviceID), errorMessage))
			return;

		device->PciBusID = strtoul(std::string{ pciBusID_s }.substr(5, 2).c_str(), NULL, 16);

		device->Name = new char[256];

		std::string devicePropName{ deviceProp.name };
		std::memcpy((void *)device->Name, deviceProp.name, devicePropName.length());

		device->ComputeMajor = deviceProp.major;
		device->ComputeMinor = deviceProp.minor;
	}

	void CudaSolver::InitializeDevice(DeviceCUDA *device, const char *errorMessage)
	{
		if (!CudaCheckError(cudaSetDevice(device->DeviceID), errorMessage))
			return;

		if (!CudaCheckError(cudaDeviceReset(), errorMessage))
			return;

		if (!CudaCheckError(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost), errorMessage))
			return;
		
		if (!CudaCheckError(cudaHostAlloc((void **)&device->SolutionCount, UINT32_LENGTH, cudaHostAllocMapped), errorMessage))
			return;

		if (!CudaCheckError(cudaHostAlloc((void **)&device->Solutions, device->MaxSolutionCount * UINT64_LENGTH, cudaHostAllocMapped), errorMessage))
			return;

		if (!CudaCheckError(cudaHostGetDevicePointer((void **)&device->SolutionCountDevice, (void *)device->SolutionCount, 0), errorMessage))
			return;

		if (!CudaCheckError(cudaHostGetDevicePointer((void **)&device->SolutionsDevice, (void *)device->Solutions, 0), errorMessage))
			return;

		device->Instance = new Device::Instance(device->DeviceID, device->PciBusID);
	}

	void CudaSolver::SetDevice(int deviceID, const char *errorMessage)
	{
		CudaCheckError(cudaSetDevice(deviceID), errorMessage);
	}

	void CudaSolver::ResetDevice(int deviceID, const char *errorMessage)
	{
		CudaCheckError(cudaDeviceReset(), errorMessage);
	}

	void CudaSolver::ReleaseDeviceObjects(DeviceCUDA *device, const char *errorMessage)
	{
		if (!CudaCheckError(cudaFreeHost(device->Solutions), errorMessage))
			return;

		if (!CudaCheckError(cudaFreeHost(device->SolutionCount), errorMessage))
			return;
	}
}