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

namespace CUDASolver
{
	namespace Device
	{
		class Instance
		{
		public:
			NV_API API;

			Instance(int deviceID, uint32_t pciBusID) :
				API { NV_API(deviceID, pciBusID) } { }
		};
	}

	struct DeviceCUDA
	{
		int DeviceID;
		int PciBusID;
		const char *Name;
		int ComputeMajor;
		int ComputeMinor;
		float Intensity;
		uint64_t Threads;
		dim3 Grid;
		dim3 Block;
		uint64_t WorkPosition;
		uint32_t MaxSolutionCount;
		uint32_t *SolutionCount;
		uint32_t *SolutionCountDevice;
		uint64_t *Solutions;
		uint64_t *SolutionsDevice;
		Device::Instance *Instance;
	};
	typedef struct DeviceCUDA DeviceCUDA;
}