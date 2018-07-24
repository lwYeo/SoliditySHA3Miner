#include "device.h"
#include "nvapi.h"

bool Device::foundNvAPI64()
{
	return NVAPI::foundNvAPI64();
}

Device::Device() :
	deviceID{ -1 },
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
}

int Device::CoreOC()
{
	int& tempDeviceID{ deviceID };
	return NVAPI::getCoreOC(tempDeviceID);
}

int Device::MemoryOC()
{
	int& tempDeviceID{ deviceID };
	return NVAPI::getMemoryOC(tempDeviceID);
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
		m_block.x = (computeVersion > 500) ? Device::MAX_TPB_500 : Device::MAX_TPB_350;
		m_lastCompute = computeVersion;
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
	return (uint64_t)((long double)hashCount.load() / (duration_cast<seconds>(steady_clock::now() - hashStartTime.load()).count()));
}
