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

		EXPORT void __CDECL__ GetDeviceCount(int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		EXPORT void __CDECL__ GetDeviceName(int deviceID, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		EXPORT CudaSolver *__CDECL__ GetInstance() noexcept;

		EXPORT void __CDECL__ DisposeInstance(CudaSolver *instance) noexcept;

		EXPORT GetKingAddressCallback __CDECL__ SetOnGetKingAddressHandler(CudaSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback __CDECL__ SetOnGetSolutionTemplateHandler(CudaSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback __CDECL__ SetOnGetWorkPositionHandler(CudaSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback __CDECL__ SetOnResetWorkPositionHandler(CudaSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback __CDECL__ SetOnIncrementWorkPositionHandler(CudaSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback __CDECL__ SetOnMessageHandler(CudaSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback __CDECL__ SetOnSolutionHandler(CudaSolver *instance, SolutionCallback solutionCallback);

		EXPORT void __CDECL__ SetSubmitStale(CudaSolver *instance, const bool submitStale);

		EXPORT void __CDECL__ AssignDevice(CudaSolver *instance, const int deviceID, unsigned int *pciBusID, float *intensity);

		EXPORT void __CDECL__ IsAssigned(CudaSolver *instance, bool *isAssigned);

		EXPORT void __CDECL__ IsAnyInitialised(CudaSolver *instance, bool *isAnyInitialised);

		EXPORT void __CDECL__ IsMining(CudaSolver *instance, bool *isMining);

		EXPORT void __CDECL__ IsPaused(CudaSolver *instance, bool *isPaused);

		EXPORT void __CDECL__ GetHashRateByDeviceID(CudaSolver *instance, const uint32_t deviceID, uint64_t *hashRate);

		EXPORT void __CDECL__ GetTotalHashRate(CudaSolver *instance, uint64_t *totalHashRate);

		EXPORT void __CDECL__ UpdatePrefix(CudaSolver *instance, const char *prefix);

		EXPORT void __CDECL__ UpdateTarget(CudaSolver *instance, const char *target);

		EXPORT void __CDECL__ PauseFinding(CudaSolver *instance, const bool pause);

		EXPORT void __CDECL__ StartFinding(CudaSolver *instance);

		EXPORT void __CDECL__ StopFinding(CudaSolver *instance);

		EXPORT void __CDECL__ GetDeviceSettingMaxCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		EXPORT void __CDECL__ GetDeviceSettingMaxMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceSettingPowerLimit(CudaSolver *instance, const int deviceID, int *powerLimit);

		EXPORT void __CDECL__ GetDeviceSettingThermalLimit(CudaSolver *instance, const int deviceID, int *thermalLimit);

		EXPORT void __CDECL__ GetDeviceSettingFanLevelPercent(CudaSolver *instance, const int deviceID, int *fanLevel);

		EXPORT void __CDECL__ GetDeviceCurrentFanTachometerRPM(CudaSolver *instance, const int deviceID, int *tachometerRPM);

		EXPORT void __CDECL__ GetDeviceCurrentTemperature(CudaSolver *instance, const int deviceID, int *temperature);

		EXPORT void __CDECL__ GetDeviceCurrentCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		EXPORT void __CDECL__ GetDeviceCurrentMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceCurrentUtilizationPercent(CudaSolver *instance, const int deviceID, int *utiliztion);

		EXPORT void __CDECL__ GetDeviceCurrentPstate(CudaSolver *instance, const int deviceID, int *pState);

		EXPORT void __CDECL__ GetDeviceCurrentThrottleReasons(CudaSolver *instance, const int deviceID, const char *throttleReasons, uint64_t *reasonSize);
	}
}

#endif // !__SOLVER__
