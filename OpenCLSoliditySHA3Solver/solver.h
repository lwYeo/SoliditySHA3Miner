#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "openCLSolver.h"

namespace OpenCLSolver
{
	extern "C"
	{
		__declspec(dllexport) void __cdecl FoundADL_API(bool *hasADL_API);

		__declspec(dllexport) void __cdecl PreInitialize(bool allowIntel, const char *errorMessge, uint64_t *errorSize);

		__declspec(dllexport) void __cdecl GetPlatformNames(const char *platformNames);

		__declspec(dllexport) void __cdecl GetDeviceCount(const char *platformName, int *deviceCount, const char *errorMessage, uint64_t *errorSize);

		__declspec(dllexport) void __cdecl GetDeviceName(const char *platformName, int deviceEnum, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize);

		__declspec(dllexport) openCLSolver *__cdecl GetInstance() noexcept;

		__declspec(dllexport) void __cdecl DisposeInstance(openCLSolver *instance) noexcept;

		__declspec(dllexport) GetKingAddressCallback __cdecl SetOnGetKingAddressHandler(openCLSolver *instance, GetKingAddressCallback getKingAddressCallback);

		__declspec(dllexport) GetSolutionTemplateCallback __cdecl SetOnGetSolutionTemplateHandler(openCLSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		__declspec(dllexport) GetWorkPositionCallback __cdecl SetOnGetWorkPositionHandler(openCLSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		__declspec(dllexport) ResetWorkPositionCallback __cdecl SetOnResetWorkPositionHandler(openCLSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		__declspec(dllexport) IncrementWorkPositionCallback __cdecl SetOnIncrementWorkPositionHandler(openCLSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		__declspec(dllexport) MessageCallback __cdecl SetOnMessageHandler(openCLSolver *instance, MessageCallback messageCallback);

		__declspec(dllexport) SolutionCallback __cdecl SetOnSolutionHandler(openCLSolver *instance, SolutionCallback solutionCallback);

		__declspec(dllexport) void __cdecl SetSubmitStale(openCLSolver *instance, const bool submitStale);

		__declspec(dllexport) void __cdecl AssignDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, float *intensity);

		__declspec(dllexport) void __cdecl IsAssigned(openCLSolver *instance, bool *isAssigned);

		__declspec(dllexport) void __cdecl IsAnyInitialised(openCLSolver *instance, bool *isAnyInitialised);

		__declspec(dllexport) void __cdecl IsMining(openCLSolver *instance, bool *isMining);

		__declspec(dllexport) void __cdecl IsPaused(openCLSolver *instance, bool *isPaused);

		__declspec(dllexport) void __cdecl GetInstanceDeviceName(openCLSolver *instance, const char *platformName, const int deviceEnum, const char *deviceName, uint64_t *nameSize);

		__declspec(dllexport) void __cdecl GetHashRateByDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, uint64_t *hashRate);

		__declspec(dllexport) void __cdecl GetTotalHashRate(openCLSolver *instance, uint64_t *totalHashRate);

		__declspec(dllexport) void __cdecl UpdatePrefix(openCLSolver *instance, const char *prefix);

		__declspec(dllexport) void __cdecl UpdateTarget(openCLSolver *instance, const char *target);

		__declspec(dllexport) void __cdecl PauseFinding(openCLSolver *instance, const bool pause);

		__declspec(dllexport) void __cdecl StartFinding(openCLSolver *instance);

		__declspec(dllexport) void __cdecl StopFinding(openCLSolver *instance);

		__declspec(dllexport) void __cdecl GetDeviceSettingMaxCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		__declspec(dllexport) void __cdecl GetDeviceSettingMaxMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		__declspec(dllexport) void __cdecl GetDeviceSettingPowerLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *powerLimit);

		__declspec(dllexport) void __cdecl GetDeviceSettingThermalLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *thermalLimit);

		__declspec(dllexport) void __cdecl GetDeviceSettingFanLevelPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *fanLevel);

		__declspec(dllexport) void __cdecl GetDeviceCurrentFanTachometerRPM(openCLSolver *instance, const char *platformName, const int deviceEnum, int *tachometerRPM);

		__declspec(dllexport) void __cdecl GetDeviceCurrentTemperature(openCLSolver *instance, const char *platformName, const int deviceEnum, int *temperature);

		__declspec(dllexport) void __cdecl GetDeviceCurrentCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock);

		__declspec(dllexport) void __cdecl GetDeviceCurrentMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock);

		__declspec(dllexport) void __cdecl GetDeviceCurrentUtilizationPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *utiliztion);
	}
}

#endif // !__SOLVER__
