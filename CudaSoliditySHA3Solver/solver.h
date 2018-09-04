#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cudaSolver.h"

#ifdef __linux__
#	define EXPORT
#	define CDECL
#else
#	define EXPORT _declspec(dllexport)
#	define CDECL __cdecl
#endif

namespace CUDASolver
{
	extern "C"
	{
		EXPORT void CDECL FoundNvAPI64(bool *hasNvAPI64);

		EXPORT void CDECL GetDeviceCount(int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		EXPORT void CDECL GetDeviceName(int deviceID, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		EXPORT CudaSolver *CDECL GetInstance() noexcept;

		EXPORT void CDECL DisposeInstance(CudaSolver *instance) noexcept;

		EXPORT GetKingAddressCallback CDECL SetOnGetKingAddressHandler(CudaSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback CDECL SetOnGetSolutionTemplateHandler(CudaSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback CDECL SetOnGetWorkPositionHandler(CudaSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback CDECL SetOnResetWorkPositionHandler(CudaSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback CDECL SetOnIncrementWorkPositionHandler(CudaSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback CDECL SetOnMessageHandler(CudaSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback CDECL SetOnSolutionHandler(CudaSolver *instance, SolutionCallback solutionCallback);

		EXPORT void CDECL SetSubmitStale(CudaSolver *instance, const bool submitStale);

		EXPORT void CDECL AssignDevice(CudaSolver *instance, const int deviceID, float *intensity);

		EXPORT void CDECL IsAssigned(CudaSolver *instance, bool *isAssigned);

		EXPORT void CDECL IsAnyInitialised(CudaSolver *instance, bool *isAnyInitialised);

		EXPORT void CDECL IsMining(CudaSolver *instance, bool *isMining);

		EXPORT void CDECL IsPaused(CudaSolver *instance, bool *isPaused);

		EXPORT void CDECL GetHashRateByDeviceID(CudaSolver *instance, const uint32_t deviceID, uint64_t *hashRate);

		EXPORT void CDECL GetTotalHashRate(CudaSolver *instance, uint64_t *totalHashRate);

		EXPORT void CDECL UpdatePrefix(CudaSolver *instance, const char *prefix);

		EXPORT void CDECL UpdateTarget(CudaSolver *instance, const char *target);

		EXPORT void CDECL PauseFinding(CudaSolver *instance, const bool pause);

		EXPORT void CDECL StartFinding(CudaSolver *instance);

		EXPORT void CDECL StopFinding(CudaSolver *instance);

		EXPORT void CDECL GetDeviceSettingMaxCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		EXPORT void CDECL GetDeviceSettingMaxMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		EXPORT void CDECL GetDeviceSettingPowerLimit(CudaSolver *instance, const int deviceID, int *powerLimit);

		EXPORT void CDECL GetDeviceSettingThermalLimit(CudaSolver *instance, const int deviceID, int *thermalLimit);

		EXPORT void CDECL GetDeviceSettingFanLevelPercent(CudaSolver *instance, const int deviceID, int *fanLevel);

		EXPORT void CDECL GetDeviceCurrentFanTachometerRPM(CudaSolver *instance, const int deviceID, int *tachometerRPM);

		EXPORT void CDECL GetDeviceCurrentTemperature(CudaSolver *instance, const int deviceID, int *temperature);

		EXPORT void CDECL GetDeviceCurrentCoreClock(CudaSolver *instance, const int deviceID, int *coreClock);

		EXPORT void CDECL GetDeviceCurrentMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock);

		EXPORT void CDECL GetDeviceCurrentUtilizationPercent(CudaSolver *instance, const int deviceID, int *utiliztion);

		EXPORT void CDECL GetDeviceCurrentPstate(CudaSolver *instance, const int deviceID, int *pState);

		EXPORT void CDECL GetDeviceCurrentThrottleReasons(CudaSolver *instance, const int deviceID, const char *throttleReasons, uint64_t *reasonSize);
	}
}

#endif // !__SOLVER__
