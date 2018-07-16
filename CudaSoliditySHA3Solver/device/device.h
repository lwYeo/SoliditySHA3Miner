#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include "..\types.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <mutex>
#	define _M_CEE 001 
#else 
#	include <mutex>
#endif 

#pragma managed(pop)

#define DEFALUT_INTENSITY 25.0F

struct Device
{
	static uint32_t constexpr MAX_TPB_500{ 1024u };
	static uint32_t constexpr MAX_TPB_350{ 384u };

public:
	int deviceID{ -1 };
	std::string name{ "" };
	uint32_t computeVersion{ 0u };
	float intensity{ DEFALUT_INTENSITY };

	bool initialized{ false };
	bool mining{ false };

	std::thread miningThread;
	std::atomic<uint64_t> hashCount{ 0ull };
	std::atomic<std::chrono::steady_clock::time_point> hashStartTime{ std::chrono::steady_clock::now() };

	uint64_t* d_Solutions;
	uint32_t* d_SolutionCount;
	uint64_t* h_Solutions{ reinterpret_cast<uint64_t *>(malloc(UINT64_LENGTH)) };
	uint32_t* h_SolutionCount{ reinterpret_cast<uint32_t *>(malloc(UINT32_LENGTH)) };

	int CoreOC();
	int MemoryOC();

	uint32_t threads();
	dim3 block();
	dim3 grid();

	uint64_t hashRate();

	static bool foundNvAPI64();

private:
	std::mutex blockMutex;
	dim3 m_block{ 1 };
	uint32_t lastCompute{ 0u };
	std::mutex gridMutex;
	dim3 m_grid{ 1 };
	uint32_t lastBlockX{ 0u };
	std::mutex hashRateMutex;
	std::mutex threadsMutex;
	uint32_t lastThreads{ 1u };
	float lastIntensity{ 0.0F };
};
