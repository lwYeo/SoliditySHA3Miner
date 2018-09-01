#include "solver.h"

namespace CUDASolver
{
	void FoundNvAPI64(bool *hasNvAPI64)
	{
		*hasNvAPI64 = CudaSolver::foundNvAPI64();
	}

	void GetDeviceCount(int *deviceCount, const char *errorMessage, uint64_t *errorSize)
	{
		CudaSolver::getDeviceCount(deviceCount, errorMessage, errorSize);
	}

	void GetDeviceName(int deviceID, const char *deviceName, uint64_t *nameSize, const char *errorMessage, uint64_t *errorSize)
	{
		CudaSolver::getDeviceName(deviceID, deviceName, nameSize, errorMessage, errorSize);
	}

	CudaSolver *GetInstance() noexcept
	{
		try { return new CudaSolver(); }
		catch (...) { return nullptr; }
	}

	void DisposeInstance(CudaSolver *instance) noexcept
	{
		try
		{
			instance->~CudaSolver();
			free(instance);
		}
		catch (...) {}
	}

	GetKingAddressCallback SetOnGetKingAddressHandler(CudaSolver *instance, GetKingAddressCallback getKingAddressCallback)
	{
		instance->m_getKingAddressCallback = getKingAddressCallback;
		return getKingAddressCallback;
	}

	GetSolutionTemplateCallback SetOnGetSolutionTemplateHandler(CudaSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback)
	{
		instance->m_getSolutionTemplateCallback = getSolutionTemplateCallback;
		return getSolutionTemplateCallback;
	}

	GetWorkPositionCallback SetOnGetWorkPositionHandler(CudaSolver *instance, GetWorkPositionCallback getWorkPositionCallback)
	{
		instance->m_getWorkPositionCallback = getWorkPositionCallback;
		return getWorkPositionCallback;
	}

	ResetWorkPositionCallback SetOnResetWorkPositionHandler(CudaSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback)
	{
		instance->m_resetWorkPositionCallback = resetWorkPositionCallback;
		return resetWorkPositionCallback;
	}

	IncrementWorkPositionCallback SetOnIncrementWorkPositionHandler(CudaSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback)
	{
		instance->m_incrementWorkPositionCallback = incrementWorkPositionCallback;
		return incrementWorkPositionCallback;
	}

	MessageCallback SetOnMessageHandler(CudaSolver *instance, MessageCallback messageCallback)
	{
		instance->m_messageCallback = messageCallback;
		return messageCallback;
	}

	SolutionCallback SetOnSolutionHandler(CudaSolver *instance, SolutionCallback solutionCallback)
	{
		instance->m_solutionCallback = solutionCallback;
		return solutionCallback;
	}

	void SetSubmitStale(CudaSolver *instance, const bool submitStale)
	{
		instance->isSubmitStale = submitStale;
	}

	void AssignDevice(CudaSolver *instance, const int deviceID, float *intensity)
	{
		instance->assignDevice(deviceID, *intensity);
	}

	void IsAssigned(CudaSolver *instance, bool *isAssigned)
	{
		*isAssigned = instance->isAssigned();
	}

	void IsAnyInitialised(CudaSolver *instance, bool *isAnyInitialised)
	{
		*isAnyInitialised = instance->isAnyInitialised();
	}

	void IsMining(CudaSolver *instance, bool *isMining)
	{
		*isMining = instance->isMining();
	}

	void IsPaused(CudaSolver *instance, bool *isPaused)
	{
		*isPaused = instance->isPaused();
	}

	void GetHashRateByDeviceID(CudaSolver *instance, const uint32_t deviceID, uint64_t *hashRate)
	{
		*hashRate = instance->getHashRateByDeviceID(deviceID);
	}

	void GetTotalHashRate(CudaSolver *instance, uint64_t *totalHashRate)
	{
		*totalHashRate = instance->getTotalHashRate();
	}

	void UpdatePrefix(CudaSolver *instance, const char *prefix)
	{
		instance->updatePrefix(prefix);
	}

	void UpdateTarget(CudaSolver *instance, const char *target)
	{
		instance->updateTarget(target);
	}

	void PauseFinding(CudaSolver *instance, const bool pause)
	{
		instance->pauseFinding(pause);
	}

	void StartFinding(CudaSolver *instance)
	{
		instance->startFinding();
	}

	void StopFinding(CudaSolver *instance)
	{
		instance->stopFinding();
	}

	void GetDeviceSettingMaxCoreClock(CudaSolver *instance, const int deviceID, int *coreClock)
	{
		*coreClock = instance->getDeviceSettingMaxCoreClock(deviceID);
	}

	void GetDeviceSettingMaxMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock)
	{
		*memoryClock = instance->getDeviceSettingMaxMemoryClock(deviceID);
	}

	void GetDeviceSettingPowerLimit(CudaSolver *instance, const int deviceID, int *powerLimit)
	{
		*powerLimit = instance->getDeviceSettingPowerLimit(deviceID);
	}

	void GetDeviceSettingThermalLimit(CudaSolver *instance, const int deviceID, int *thermalLimit)
	{
		*thermalLimit = instance->getDeviceSettingThermalLimit(deviceID);
	}

	void GetDeviceSettingFanLevelPercent(CudaSolver *instance, const int deviceID, int *fanLevel)
	{
		*fanLevel = instance->getDeviceSettingFanLevelPercent(deviceID);
	}

	void GetDeviceCurrentFanTachometerRPM(CudaSolver *instance, const int deviceID, int *tachometerRPM)
	{
		*tachometerRPM = instance->getDeviceCurrentFanTachometerRPM(deviceID);
	}

	void GetDeviceCurrentTemperature(CudaSolver *instance, const int deviceID, int *temperature)
	{
		*temperature = instance->getDeviceCurrentTemperature(deviceID);
	}

	void GetDeviceCurrentCoreClock(CudaSolver *instance, const int deviceID, int *coreClock)
	{
		*coreClock = instance->getDeviceCurrentCoreClock(deviceID);
	}

	void GetDeviceCurrentMemoryClock(CudaSolver *instance, const int deviceID, int *memoryClock)
	{
		*memoryClock = instance->getDeviceCurrentMemoryClock(deviceID);
	}

	void GetDeviceCurrentUtilizationPercent(CudaSolver *instance, const int deviceID, int *utiliztion)
	{
		*utiliztion = instance->getDeviceCurrentUtilizationPercent(deviceID);
	}

	void GetDeviceCurrentPstate(CudaSolver *instance, const int deviceID, int *pState)
	{
		*pState = instance->getDeviceCurrentPstate(deviceID);
	}

	void GetDeviceCurrentThrottleReasons(CudaSolver *instance, const int deviceID, const char *throttleReasons, uint64_t *reasonSize)
	{
		auto reasons = instance->getDeviceCurrentThrottleReasons(deviceID);
		const char *reasonStr = reasons.c_str();
		std::memcpy((void *)throttleReasons, reasonStr, reasons.length());
		std::memset((void *)&throttleReasons[reasons.length()], '\0', 1ull);
		*reasonSize = reasons.length();
	}
}