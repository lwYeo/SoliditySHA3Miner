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

#include "solver.h"

namespace CUDASolver
{
	void FoundNvAPI64(bool *hasNvAPI64)
	{
		*hasNvAPI64 = CudaSolver::FoundNvAPI64();
	}

	void GetDeviceCount(int *deviceCount, const char *errorMessage)
	{
		CudaSolver::GetDeviceCount(deviceCount, errorMessage);
	}

	void GetDeviceName(int deviceID, const char *deviceName, const char *errorMessage)
	{
		CudaSolver::GetDeviceName(deviceID, deviceName, errorMessage);
	}

	CudaSolver *GetInstance() noexcept
	{
		return new CudaSolver();
	}

	void DisposeInstance(CudaSolver *instance) noexcept
	{
		delete instance;
	}

	void GetDeviceProperties(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage)
	{
		instance->GetDeviceProperties(device, errorMessage);
	}

	void InitializeDevice(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage)
	{
		instance->InitializeDevice(device, errorMessage);
	}

	void SetDevice(CudaSolver *instance, int deviceID, const char *errorMessage)
	{
		instance->SetDevice(deviceID, errorMessage);
	}

	void ResetDevice(CudaSolver *instance, int deviceID, const char *errorMessage)
	{
		instance->ResetDevice(deviceID, errorMessage);
	}

	void ReleaseDeviceObjects(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage)
	{
		instance->ReleaseDeviceObjects(device, errorMessage);
	}

	void PushHigh64Target(CudaSolver *instance, uint64_t *high64Target, const char *errorMessage)
	{
		instance->PushHigh64Target(high64Target, errorMessage);
	}

	void PushMidState(CudaSolver *instance, sponge_ut *midState, const char *errorMessage)
	{
		instance->PushMidState(midState, errorMessage);
	}

	void PushTarget(CudaSolver *instance, byte32_t *target, const char *errorMessage)
	{
		instance->PushTarget(target, errorMessage);
	}

	void PushMessage(CudaSolver *instance, message_ut *message, const char *errorMessage)
	{
		instance->PushMessage(message, errorMessage);
	}

	void HashMidState(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage)
	{
		instance->HashMidState(device, errorMessage);
	}

	void HashMessage(CudaSolver *instance, DeviceCUDA *device, const char *errorMessage)
	{
		instance->HashMessage(device, errorMessage);
	}

	void GetDeviceSettingMaxCoreClock(DeviceCUDA device, int *coreClock)
	{
		*coreClock = -1;
		device.Instance->API.getSettingMaxCoreClock(coreClock);
	}

	void GetDeviceSettingMaxMemoryClock(DeviceCUDA device, int *memoryClock)
	{
		*memoryClock = -1;
		device.Instance->API.getSettingMaxMemoryClock(memoryClock);
	}

	void GetDeviceSettingPowerLimit(DeviceCUDA device, int *powerLimit)
	{
		*powerLimit = -1;
		device.Instance->API.getSettingPowerLimit(powerLimit);
	}

	void GetDeviceSettingThermalLimit(DeviceCUDA device, int *thermalLimit)
	{
		*thermalLimit = INT32_MIN;
		device.Instance->API.getSettingThermalLimit(thermalLimit);
	}

	void GetDeviceSettingFanLevelPercent(DeviceCUDA device, int *fanLevel)
	{
		*fanLevel = -1;
		device.Instance->API.getSettingFanLevelPercent(fanLevel);
	}

	void GetDeviceCurrentFanTachometerRPM(DeviceCUDA device, int *tachometerRPM)
	{
		*tachometerRPM = -1;
		device.Instance->API.getCurrentFanTachometerRPM(tachometerRPM);
	}

	void GetDeviceCurrentTemperature(DeviceCUDA device, int *temperature)
	{
		*temperature = INT32_MIN;
		device.Instance->API.getCurrentTemperature(temperature);
	}

	void GetDeviceCurrentCoreClock(DeviceCUDA device, int *coreClock)
	{
		*coreClock = -1;
		device.Instance->API.getCurrentCoreClock(coreClock);
	}

	void GetDeviceCurrentMemoryClock(DeviceCUDA device, int *memoryClock)
	{
		*memoryClock = -1;
		device.Instance->API.getCurrentMemoryClock(memoryClock);
	}

	void GetDeviceCurrentUtilizationPercent(DeviceCUDA device, int *utiliztion)
	{
		*utiliztion = -1;
		device.Instance->API.getCurrentUtilizationPercent(utiliztion);
	}

	void GetDeviceCurrentPstate(DeviceCUDA device, int *pState)
	{
		*pState = -1;
		device.Instance->API.getCurrentPstate(pState);
	}

	void GetDeviceCurrentThrottleReasons(DeviceCUDA device, const char *throttleReasons)
	{
		device.Instance->API.getCurrentThrottleReasons(throttleReasons);
	}
}