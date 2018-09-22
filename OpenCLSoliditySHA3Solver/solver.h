#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "openCLSolver.h"

#ifdef __linux__
#	define EXPORT
#	define __CDECL__
#else
#	define EXPORT _declspec(dllexport)
#	define __CDECL__ __cdecl
#endif

namespace OpenCLSolver
{
	extern "C"
	{
		EXPORT void __CDECL__ FoundADL_API(bool *hasADL_API);

		EXPORT void __CDECL__ PreInitialize(bool allowIntel, const char *errorMessge, uint64_t *errorSize);

		EXPORT void __CDECL__ GetPlatformNames(const char *platformNames);

		EXPORT void __CDECL__ GetDeviceCount(const char *platformName, int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		EXPORT void __CDECL__ GetDeviceName(const char *platformName, int deviceEnum, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		EXPORT openCLSolver *__CDECL__ GetInstance() noexcept;

		EXPORT void __CDECL__ DisposeInstance(openCLSolver *instance) noexcept;

		EXPORT GetKingAddressCallback __CDECL__ SetOnGetKingAddressHandler(openCLSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback __CDECL__ SetOnGetSolutionTemplateHandler(openCLSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback __CDECL__ SetOnGetWorkPositionHandler(openCLSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback __CDECL__ SetOnResetWorkPositionHandler(openCLSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback __CDECL__ SetOnIncrementWorkPositionHandler(openCLSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback __CDECL__ SetOnMessageHandler(openCLSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback __CDECL__ SetOnSolutionHandler(openCLSolver *instance, SolutionCallback solutionCallback);

		EXPORT void __CDECL__ SetSubmitStale(openCLSolver *instance, const bool submitStale);

		EXPORT void __CDECL__ AssignDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, float *intensity);

		EXPORT void __CDECL__ IsAssigned(openCLSolver *instance, bool *isAssigned);

		EXPORT void __CDECL__ IsAnyInitialised(openCLSolver *instance, bool *isAnyInitialised);

		EXPORT void __CDECL__ IsMining(openCLSolver *instance, bool *isMining);

		EXPORT void __CDECL__ IsPaused(openCLSolver *instance, bool *isPaused);

		EXPORT void __CDECL__ GetInstanceDeviceName(openCLSolver *instance, const char *platformName, const int deviceEnum, const char *deviceName, uint64_t *nameSize);

		EXPORT void __CDECL__ GetHashRateByDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, uint64_t *hashRate);

		EXPORT void __CDECL__ GetTotalHashRate(openCLSolver *instance, uint64_t *totalHashRate);

		EXPORT void __CDECL__ UpdatePrefix(openCLSolver *instance, const char *prefix);

		EXPORT void __CDECL__ UpdateTarget(openCLSolver *instance, const char *target);

		EXPORT void __CDECL__ PauseFinding(openCLSolver *instance, const bool pause);

		EXPORT void __CDECL__ StartFinding(openCLSolver *instance);

		EXPORT void __CDECL__ StopFinding(openCLSolver *instance);

		EXPORT void __CDECL__ GetDeviceSettingMaxCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		EXPORT void __CDECL__ GetDeviceSettingMaxMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceSettingPowerLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *powerLimit);

		EXPORT void __CDECL__ GetDeviceSettingThermalLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *thermalLimit);

		EXPORT void __CDECL__ GetDeviceSettingFanLevelPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *fanLevel);

		EXPORT void __CDECL__ GetDeviceCurrentFanTachometerRPM(openCLSolver *instance, const char *platformName, const int deviceEnum, int *tachometerRPM);

		EXPORT void __CDECL__ GetDeviceCurrentTemperature(openCLSolver *instance, const char *platformName, const int deviceEnum, int *temperature);

		EXPORT void __CDECL__ GetDeviceCurrentCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		EXPORT void __CDECL__ GetDeviceCurrentMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		EXPORT void __CDECL__ GetDeviceCurrentUtilizationPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *utiliztion);
	}
}

#endif // !__SOLVER__
