#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cudaSolver.h"

namespace CUDASolver
{
	extern "C"
	{
		__declspec(dllexport) void __cdecl FoundNvAPI64(bool *hasNvAPI64);

		__declspec(dllexport) void __cdecl GetDeviceCount(int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		__declspec(dllexport) void __cdecl GetDeviceName(int deviceID, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		__declspec(dllexport) CudaSolver *__cdecl GetInstance() noexcept;

		__declspec(dllexport) void __cdecl DisposeInstance(CudaSolver *instance) noexcept;

		__declspec(dllexport) GetKingAddressCallback __cdecl SetOnGetKingAddressHandler(CudaSolver *instance, GetKingAddressCallback getKingAddressCallback);

		__declspec(dllexport) GetSolutionTemplateCallback __cdecl SetOnGetSolutionTemplateHandler(CudaSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		__declspec(dllexport) GetWorkPositionCallback __cdecl SetOnGetWorkPositionHandler(CudaSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		__declspec(dllexport) ResetWorkPositionCallback __cdecl SetOnResetWorkPositionHandler(CudaSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		__declspec(dllexport) IncrementWorkPositionCallback __cdecl SetOnIncrementWorkPositionHandler(CudaSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		__declspec(dllexport) MessageCallback __cdecl SetOnMessageHandler(CudaSolver *instance, MessageCallback messageCallback);

		__declspec(dllexport) SolutionCallback __cdecl SetOnSolutionHandler(CudaSolver *instance, SolutionCallback solutionCallback);

		__declspec(dllexport) void __cdecl SetSubmitStale(CudaSolver *instance, const bool submitStale);

		__declspec(dllexport) void __cdecl AssignDevice(CudaSolver *instance, const int deviceID, float *intensity);

		__declspec(dllexport) void __cdecl IsAssigned(CudaSolver *instance, bool *isAssigned);

		__declspec(dllexport) void __cdecl IsAnyInitialised(CudaSolver *instance, bool *isAnyInitialised);

		__declspec(dllexport) void __cdecl IsMining(CudaSolver *instance, bool *isMining);

		__declspec(dllexport) void __cdecl IsPaused(CudaSolver *instance, bool *isPaused);

		__declspec(dllexport) void __cdecl GetHashRateByDeviceID(CudaSolver *instance, const uint32_t deviceID, uint64_t *hashRate);

		__declspec(dllexport) void __cdecl GetTotalHashRate(CudaSolver *instance, uint64_t *totalHashRate);

		__declspec(dllexport) void __cdecl UpdatePrefix(CudaSolver *instance, const char *prefix);

		__declspec(dllexport) void __cdecl UpdateTarget(CudaSolver *instance, const char *target);

		__declspec(dllexport) void __cdecl PauseFinding(CudaSolver *instance, const bool pause);

		__declspec(dllexport) void __cdecl StartFinding(CudaSolver *instance);

		__declspec(dllexport) void __cdecl StopFinding(CudaSolver *instance);

		__declspec(dllexport) void __cdecl GetDeviceSettingMaxCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		__declspec(dllexport) void __cdecl GetDeviceSettingMaxMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		__declspec(dllexport) void __cdecl GetDeviceSettingPowerLimit(CudaSolver *instance, const int deviceID, int *powerLimit);

		__declspec(dllexport) void __cdecl GetDeviceSettingThermalLimit(CudaSolver *instance, const int deviceID, int *thermalLimit);

		__declspec(dllexport) void __cdecl GetDeviceSettingFanLevelPercent(CudaSolver *instance, const int deviceID, int *fanLevel);

		__declspec(dllexport) void __cdecl GetDeviceCurrentFanTachometerRPM(CudaSolver *instance, const int deviceID, int *tachometerRPM);

		__declspec(dllexport) void __cdecl GetDeviceCurrentTemperature(CudaSolver *instance, const int deviceID, int *temperature);

		__declspec(dllexport) void __cdecl GetDeviceCurrentCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		__declspec(dllexport) void __cdecl GetDeviceCurrentMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		__declspec(dllexport) void __cdecl GetDeviceCurrentUtilizationPercent(CudaSolver *instance, const int deviceID, int *utiliztion);

		__declspec(dllexport) void __cdecl GetDeviceCurrentPstate(CudaSolver *instance, const int deviceID, int *pState);

		__declspec(dllexport) void __cdecl GetDeviceCurrentThrottleReasons(CudaSolver *instance, const int deviceID, const char *throttleReasons, uint64_t *reasonSize);
	}
}

#endif // !__SOLVER__
