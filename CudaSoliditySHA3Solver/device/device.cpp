#include "device.h"
#include <string>

namespace CUDASolver
{
	// --------------------------------------------------------------------
	// Public
	// --------------------------------------------------------------------

	Device::Device(int deviceID) :
		deviceID{ deviceID },
		name{ "" },
		computeVersion{ 0u },
		intensity{ DEFALUT_INTENSITY },
		initialized{ false },
		mining{ false },
		hashCount{ 0ull },
		hashStartTime{ std::chrono::steady_clock::now() },
		m_block{ 1u },
		m_lastCompute{ 0u },
		m_grid{ 1u },
		m_lastBlockX{ 0u },
		m_lastThreads{ 1u },
		m_lastIntensity{ 0.0F }
	{
		char pciBusID_s[13];
		if (cudaDeviceGetPCIBusId(pciBusID_s, 13, deviceID) == (cudaError_t)NVAPI_OK)
		{
			pciBusID = strtoul(std::string{ pciBusID_s }.substr(5, 2).c_str(), NULL, 16);
			m_api.assignPciBusID(pciBusID);
		}
	}
	
	uint32_t Device::getPciBusID()
	{
		return (uint32_t)m_api.deviceBusID;
	}

	bool Device::getSettingMaxCoreClock(int *maxCoreClock, std::string *errorMessage)
	{
		auto status = m_api.getSettingMaxCoreClock(maxCoreClock);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getSettingMaxMemoryClock(int *maxMemoryClock, std::string *errorMessage)
	{
		auto status = m_api.getSettingMaxMemoryClock(maxMemoryClock);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getSettingPowerLimit(int *powerLimit, std::string *errorMessage)
	{
		auto status = m_api.getSettingPowerLimit(powerLimit);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getSettingThermalLimit(int *thermalLimit, std::string *errorMessage)
	{
		auto status = m_api.getSettingThermalLimit(thermalLimit);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getSettingFanLevelPercent(int *fanLevel, std::string *errorMessage)
	{
		auto status = m_api.getSettingFanLevelPercent(fanLevel);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentFanTachometerRPM(int *tachometerRPM, std::string *errorMessage)
	{
		auto status = m_api.getCurrentFanTachometerRPM(tachometerRPM);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentTemperature(int *temperature, std::string *errorMessage)
	{
		auto status = m_api.getCurrentTemperature(temperature);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentCoreClock(int *coreClock, std::string *errorMessage)
	{
		auto status = m_api.getCurrentCoreClock(coreClock);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentMemoryClock(int *memoryClock, std::string *errorMessage)
	{
		auto status = m_api.getCurrentMemoryClock(memoryClock);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentUtilizationPercent(int *utilization, std::string *errorMessage)
	{
		auto status = m_api.getCurrentUtilizationPercent(utilization);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentPstate(int *pstate, std::string *errorMessage)
	{
		auto status = m_api.getCurrentPstate(pstate);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	bool Device::getCurrentThrottleReasons(std::string *reasons, std::string *errorMessage)
	{
		auto status = m_api.getCurrentThrottleReasons(reasons);

		if (status != NVAPI_OK)
			m_api.getErrorMessage(status, errorMessage);

		return (status == NVAPI_OK);
	}

	uint32_t Device::threads()
	{
		if (computeVersion <= 500) intensity = intensity <= 40.55F ? intensity : 40.55F;

		if (intensity != m_lastIntensity)
		{
			m_lastThreads = (uint32_t)std::pow(2, intensity);
			m_lastIntensity = intensity;
			m_lastBlockX = 0u;
		}
		return m_lastThreads;
	}

	dim3 Device::block()
	{
		if (m_lastCompute != computeVersion)
		{
			m_lastCompute = computeVersion;
			switch (computeVersion)
			{
			case 520:
			case 610:
			case 700:
			case 720:
			case 750:
				m_block.x = 1024u;
				break;
			case 300:
			case 320:
			case 350:
			case 370:
			case 500:
			case 530:
			case 600:
			case 620:
			default:
				m_block.x = (computeVersion >= 800) ? 1024u : 384u;
				break;
			}
		}
		return m_block;
	}

	dim3 Device::grid()
	{
		if (m_lastBlockX != block().x)
		{
			m_grid.x = uint32_t((threads() + block().x - 1) / block().x);
			m_lastBlockX = block().x;
		}
		return m_grid;
	}

	uint64_t Device::hashRate()
	{
		using namespace std::chrono;
		return (uint64_t)((long double)hashCount.load() / (duration_cast<seconds>(steady_clock::now() - hashStartTime).count()));
	}
}