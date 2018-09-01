#include "solver.h"

namespace OpenCLSolver
{
	void FoundADL_API(bool *hasADL_API)
	{
		*hasADL_API = openCLSolver::foundAdlApi();
	}

	void PreInitialize(bool allowIntel, const char *errorMessge, uint64_t *errorSize)
	{
		std::string errMsg{ 0 };
		openCLSolver::preInitialize(allowIntel, errMsg);

		errorMessge = errMsg.c_str();
		*errorSize = errMsg.length();
	}

	void GetPlatformNames(const char *platformNames)
	{
		platformNames = openCLSolver::getPlatformNames().c_str();
	}

	void GetDeviceCount(const char *platformName, int *deviceCount, const char *errorMessage, uint64_t *errorSize)
	{
		std::string errMsg{ 0 };
		*deviceCount = openCLSolver::getDeviceCount(platformName, errMsg);

		errorMessage = errMsg.c_str();
		*errorSize = errMsg.length();
	}

	void GetDeviceName(const char *platformName, int deviceEnum, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize)
	{
		std::string errMsg{ 0 };
		auto devName = openCLSolver::getDeviceName(platformName, deviceEnum, errMsg);
		deviceName = devName.c_str();

		errorMessage = errMsg.c_str();
		*errorSize = errMsg.length();
	}

	openCLSolver *GetInstance() noexcept
	{
		try { return new openCLSolver(); }
		catch (...) { return nullptr; }
	}

	void DisposeInstance(openCLSolver *instance) noexcept
	{
		try
		{
			instance->~openCLSolver();
			free(instance);
		}
		catch (...) {}
	}

	GetKingAddressCallback SetOnGetKingAddressHandler(openCLSolver *instance, GetKingAddressCallback getKingAddressCallback)
	{
		instance->m_getKingAddressCallback = getKingAddressCallback;
		return getKingAddressCallback;
	}

	GetSolutionTemplateCallback SetOnGetSolutionTemplateHandler(openCLSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback)
	{
		instance->m_getSolutionTemplateCallback = getSolutionTemplateCallback;
		return getSolutionTemplateCallback;
	}

	GetWorkPositionCallback SetOnGetWorkPositionHandler(openCLSolver *instance, GetWorkPositionCallback getWorkPositionCallback)
	{
		instance->m_getWorkPositionCallback = getWorkPositionCallback;
		return getWorkPositionCallback;
	}

	ResetWorkPositionCallback SetOnResetWorkPositionHandler(openCLSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback)
	{
		instance->m_resetWorkPositionCallback = resetWorkPositionCallback;
		return resetWorkPositionCallback;
	}

	IncrementWorkPositionCallback SetOnIncrementWorkPositionHandler(openCLSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback)
	{
		instance->m_incrementWorkPositionCallback = incrementWorkPositionCallback;
		return incrementWorkPositionCallback;
	}

	MessageCallback SetOnMessageHandler(openCLSolver *instance, MessageCallback messageCallback)
	{
		instance->m_messageCallback = messageCallback;
		return messageCallback;
	}

	SolutionCallback SetOnSolutionHandler(openCLSolver *instance, SolutionCallback solutionCallback)
	{
		instance->m_solutionCallback = solutionCallback;
		return solutionCallback;
	}

	void SetSubmitStale(openCLSolver *instance, const bool submitStale)
	{
		instance->isSubmitStale = submitStale;
	}

	void AssignDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, float *intensity)
	{
		instance->assignDevice(platformName, deviceEnum, *intensity);
	}

	void IsAssigned(openCLSolver *instance, bool *isAssigned)
	{
		*isAssigned = instance->isAssigned();
	}

	void IsAnyInitialised(openCLSolver *instance, bool *isAnyInitialised)
	{
		*isAnyInitialised = instance->isAnyInitialised();
	}

	void IsMining(openCLSolver *instance, bool *isMining)
	{
		*isMining = instance->isMining();
	}

	void IsPaused(openCLSolver *instance, bool *isPaused)
	{
		*isPaused = instance->isPaused();
	}

	void GetInstanceDeviceName(openCLSolver *instance, const char *platformName, const int deviceEnum, const char *deviceName, uint64_t *nameSize)
	{
		auto devName = instance->getDeviceName(platformName, deviceEnum);
		const char *deviceNameStr = devName.c_str();

		std::memcpy((void *)deviceName, deviceNameStr, devName.length());
		std::memset((void *)&deviceName[devName.length()], '\0', 1ull);
		*nameSize = devName.length();
	}

	void GetHashRateByDevice(openCLSolver *instance, const char *platformName, const int deviceEnum, uint64_t *hashRate)
	{
		*hashRate = instance->getHashRateByDevice(platformName, deviceEnum);
	}

	void GetTotalHashRate(openCLSolver *instance, uint64_t *totalHashRate)
	{
		*totalHashRate = instance->getTotalHashRate();
	}

	void UpdatePrefix(openCLSolver *instance, const char *prefix)
	{
		instance->updatePrefix(prefix);
	}

	void UpdateTarget(openCLSolver *instance, const char *target)
	{
		instance->updateTarget(target);
	}

	void PauseFinding(openCLSolver *instance, const bool pause)
	{
		instance->pauseFinding(pause);
	}

	void StartFinding(openCLSolver *instance)
	{
		instance->startFinding();
	}

	void StopFinding(openCLSolver *instance)
	{
		instance->stopFinding();
	}

	void GetDeviceSettingMaxCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock)
	{
		*coreClock = instance->getDeviceSettingMaxCoreClock(platformName, deviceEnum);
	}

	void GetDeviceSettingMaxMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock)
	{
		*memoryClock = instance->getDeviceSettingMaxMemoryClock(platformName, deviceEnum);
	}

	void GetDeviceSettingPowerLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *powerLimit)
	{
		*powerLimit = instance->getDeviceSettingPowerLimit(platformName, deviceEnum);
	}

	void GetDeviceSettingThermalLimit(openCLSolver *instance, const char *platformName, const int deviceEnum, int *thermalLimit)
	{
		*thermalLimit = instance->getDeviceSettingThermalLimit(platformName, deviceEnum);
	}

	void GetDeviceSettingFanLevelPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *fanLevel)
	{
		*fanLevel = instance->getDeviceSettingFanLevelPercent(platformName, deviceEnum);
	}

	void GetDeviceCurrentFanTachometerRPM(openCLSolver *instance, const char *platformName, const int deviceEnum, int *tachometerRPM)
	{
		*tachometerRPM = instance->getDeviceCurrentFanTachometerRPM(platformName, deviceEnum);
	}

	void GetDeviceCurrentTemperature(openCLSolver *instance, const char *platformName, const int deviceEnum, int *temperature)
	{
		*temperature = instance->getDeviceCurrentTemperature(platformName, deviceEnum);
	}

	void GetDeviceCurrentCoreClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *coreClock)
	{
		*coreClock = instance->getDeviceCurrentCoreClock(platformName, deviceEnum);
	}

	void GetDeviceCurrentMemoryClock(openCLSolver *instance, const char *platformName, const int deviceEnum, int *memoryClock)
	{
		*memoryClock = instance->getDeviceCurrentMemoryClock(platformName, deviceEnum);
	}

	void GetDeviceCurrentUtilizationPercent(openCLSolver *instance, const char *platformName, const int deviceEnum, int *utiliztion)
	{
		*utiliztion = instance->getDeviceCurrentUtilizationPercent(platformName, deviceEnum);
	}
}