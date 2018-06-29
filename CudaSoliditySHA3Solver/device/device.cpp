#include "device.h"
#include "nvapi.h"

int Device::CoreOC()
{
	int tempDeviceID{ deviceID };
	return NVAPI::getCoreOC(tempDeviceID);
}

int Device::MemoryOC()
{
	int tempDeviceID{ deviceID };
	return NVAPI::getMemoryOC(tempDeviceID);
}

uint32_t Device::threads()
{
	std::lock_guard<std::mutex> lock(threadsMutex);

	if (computeVersion <= 500) intensity = intensity <= 40.55F ? intensity : 40.55F;

	if (intensity != lastIntensity)
	{
		lastThreads = (uint32_t)std::pow(2, intensity);
		lastIntensity = intensity;
		lastBlockX = 0u;
	}
	return lastThreads;
}

dim3 Device::block()
{
	std::lock_guard<std::mutex> lock(blockMutex);

	if (lastCompute != computeVersion)
	{
		m_block.x = (computeVersion > 500) ? Device::MAX_TPB_500 : Device::MAX_TPB_350;
		lastCompute = computeVersion;
	}
	return m_block;
}

dim3 Device::grid()
{
	std::lock_guard<std::mutex> lock(gridMutex);

	if (lastBlockX != block().x)
	{
		m_grid.x = uint32_t((threads() + block().x - 1) / block().x);
		lastBlockX = block().x;
	}
	return m_grid;
}

uint64_t Device::hashRate()
{
	std::lock_guard<std::mutex> lock(hashRateMutex);
	using namespace std::chrono;

	long double dHashRate{ (long double)hashCount.load() / (duration_cast<seconds>(steady_clock::now() - hashStartTime.load()).count()) };
	return (uint64_t)dHashRate;;
}

bool Device::foundNvAPI64()
{
	return NVAPI::foundNvAPI64();
}
