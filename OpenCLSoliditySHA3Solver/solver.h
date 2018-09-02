#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "openCLSolver.h"

#ifdef __linux__
#	define EXPORT
#	define CDECL
#else
#	define EXPORT _declspec(dllexport)
#	define CDECL __cdecl
#endif

namespace OpenCLSolver
{
	extern "C"
	{
		EXPORT void CDECL FoundADL_API(bool *hasADL_API);

		EXPORT void CDECL PreInitialize(bool allowIntel, const char *errorMessge, uint64_t *errorSize);

		EXPORT void CDECL GetPlatformNames(const char *platformNames);

		EXPORT void CDECL GetDeviceCount(const char *platformName, int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		EXPORT void CDECL GetDeviceName(const char *platformName, int deviceEnum, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		EXPORT openCLSolver *CDECL GetInstance() noexcept;

		EXPORT void CDECL DisposeInstance(openCLSolver *instance) noexcept;

		EXPORT GetKingAddressCallback CDECL SetOnGetKingAddressHandler(openCLSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback CDECL SetOnGetSolutionTemplateHandler(openCLSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback CDECL SetOnGetWorkPositionHandler(openCLSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback CDECL SetOnResetWorkPositionHandler(openCLSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback CDECL SetOnIncrementWorkPositionHandler(openCLSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback CDECL SetOnMessageHandler(openCLSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback CDECL SetOnSolutionHandler(openCLSolver *instance, SolutionCallback solutionCallback);

		EXPORT void CDECL SetSubmitStale(openCLSolver *instance, const bool submitStale);

		EXPORT void CDECL AssignDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, float *intensity);

		EXPORT void CDECL IsAssigned(openCLSolver *instance, bool *isAssigned);

		EXPORT void CDECL IsAnyInitialised(openCLSolver *instance, bool *isAnyInitialised);

		EXPORT void CDECL IsMining(openCLSolver *instance, bool *isMining);

		EXPORT void CDECL IsPaused(openCLSolver *instance, bool *isPaused);

		EXPORT void CDECL GetInstanceDeviceName(openCLSolver *instance, const char *platformName, const int deviceEnum, const char *deviceName, uint64_t *nameSize);

		EXPORT void CDECL GetHashRateByDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, uint64_t *hashRate);

		EXPORT void CDECL GetTotalHashRate(openCLSolver *instance, uint64_t *totalHashRate);

		EXPORT void CDECL UpdatePrefix(openCLSolver *instance, const char *prefix);

		EXPORT void CDECL UpdateTarget(openCLSolver *instance, const char *target);

		EXPORT void CDECL PauseFinding(openCLSolver *instance, const bool pause);

		EXPORT void CDECL StartFinding(openCLSolver *instance);

		EXPORT void CDECL StopFinding(openCLSolver *instance);

		EXPORT void CDECL GetDeviceSettingMaxCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		EXPORT void CDECL GetDeviceSettingMaxMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		EXPORT void CDECL GetDeviceSettingPowerLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *powerLimit);

		EXPORT void CDECL GetDeviceSettingThermalLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *thermalLimit);

		EXPORT void CDECL GetDeviceSettingFanLevelPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *fanLevel);

		EXPORT void CDECL GetDeviceCurrentFanTachometerRPM(openCLSolver *instance, const char *platformName, const int deviceEnum, int *tachometerRPM);

		EXPORT void CDECL GetDeviceCurrentTemperature(openCLSolver *instance, const char *platformName, const int deviceEnum, int *temperature);

		EXPORT void CDECL GetDeviceCurrentCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		EXPORT void CDECL GetDeviceCurrentMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		EXPORT void CDECL GetDeviceCurrentUtilizationPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *utiliztion);
	}
}

#endif // !__SOLVER__
