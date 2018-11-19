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

#ifndef __SOLVER__
#define __SOLVER__

#include "cudaSolver.h"

#ifdef __linux__
#	define EXPORT
#	define __CDECL__
#else
#	define EXPORT _declspec(dllexport)
#	define __CDECL__ __cdecl
#endif

namespace CUDASolver
{
	extern "C"
	{
		EXPORT void __CDECL__ FoundNvAPI64(bool *hasNvAPI64);

		EXPORT void __CDECL__ GetDeviceCount(int *deviceCount, const char *errorMessage);

		EXPORT void __CDECL__ GetDeviceName(int deviceID, const char *deviceName, const char *errorMessage);

		EXPORT CudaSolver *__CDECL__ GetInstance() noexcept;

		EXPORT void __CDECL__ DisposeInstance(CudaSolver *instance) noexcept;

		EXPORT void __CDECL__ GetDeviceProperties(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage);

		EXPORT void __CDECL__ InitializeDevice(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage);

		EXPORT void __CDECL__ SetDevice(CudaSolver *instance, int deviceID, const char *errorMessage);

		EXPORT void __CDECL__ ResetDevice(CudaSolver *instance, int deviceID, const char *errorMessage);

		EXPORT void __CDECL__ ReleaseDeviceObjects(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage);

		EXPORT void __CDECL__ PushHigh64Target(CudaSolver *instance, uint64_t *high64Target, const char *errorMessage);

		EXPORT void __CDECL__ PushMidState(CudaSolver *instance, sponge_ut *midState, const char *errorMessage);

		EXPORT void __CDECL__ PushTarget(CudaSolver *instance, byte32_t *high64Target, const char *errorMessage);

		EXPORT void __CDECL__ PushMessage(CudaSolver *instance, message_ut *midState, const char *errorMessage);

		EXPORT void __CDECL__ HashMidState(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage);

		EXPORT void __CDECL__ HashMessage(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage);

		EXPORT void __CDECL__ GetDeviceSettingMaxCoreClock(DeviceCUDA device, int *coreClock);

		EXPORT void __CDECL__ GetDeviceSettingMaxMemoryClock(DeviceCUDA device, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceSettingPowerLimit(DeviceCUDA device, int *powerLimit);

		EXPORT void __CDECL__ GetDeviceSettingThermalLimit(DeviceCUDA device, int *thermalLimit);

		EXPORT void __CDECL__ GetDeviceSettingFanLevelPercent(DeviceCUDA device, int *fanLevel);

		EXPORT void __CDECL__ GetDeviceCurrentFanTachometerRPM(DeviceCUDA device, int *tachometerRPM);

		EXPORT void __CDECL__ GetDeviceCurrentTemperature(DeviceCUDA device, int *temperature);

		EXPORT void __CDECL__ GetDeviceCurrentCoreClock(DeviceCUDA device, int *coreClock);

		EXPORT void __CDECL__ GetDeviceCurrentMemoryClock(DeviceCUDA device, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceCurrentUtilizationPercent(DeviceCUDA device, int *utiliztion);

		EXPORT void __CDECL__ GetDeviceCurrentPstate(DeviceCUDA device, int *pState);

		EXPORT void __CDECL__ GetDeviceCurrentThrottleReasons(DeviceCUDA device, const char *throttleReasons);
	}
}

#endif // !__SOLVER__