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

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <device_launch_parameters.h>
#include "device/nv_api.h"
#include "device/instance.h"

// --------------------------------------------------------------------
// CUDA common constants
// --------------------------------------------------------------------

__constant__ static uint64_t const Keccak_f1600_RC[24] =
{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

namespace CUDASolver
{
	class CudaSolver
	{
	public:
		static bool FoundNvAPI64();

		static void GetDeviceCount(int *deviceCount, const char *errorMessage);

		static void GetDeviceName(int deviceID, const char *deviceName, const char *errorMessage);

		CudaSolver() noexcept;
		~CudaSolver() noexcept;

		void GetDeviceProperties(DeviceCUDA *device, const char *errorMessage);

		void InitializeDevice(DeviceCUDA *device, const char *errorMessage);

		void SetDevice(int deviceID, const char *errorMessage);
		void ResetDevice(int deviceID, const char *errorMessage);
		void ReleaseDeviceObjects(DeviceCUDA *device, const char *errorMessage);

		void PushHigh64Target(uint64_t *high64Target, const char *errorMessage);
		void PushMidState(sponge_ut *midState, const char *errorMessage);
		void PushTarget(byte32_t *target, const char *errorMessage);
		void PushMessage(message_ut *message, const char *errorMessage);

		void HashMidState(DeviceCUDA *device, const char *errorMessage);
		void HashMessage(DeviceCUDA *device, const char *errorMessage);
	};
}