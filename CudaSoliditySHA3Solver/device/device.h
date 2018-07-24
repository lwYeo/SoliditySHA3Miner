#pragma once

#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include "..\types.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <thread>
#	define _M_CEE 001 
#else 
#	include <thread>
#endif 

#pragma managed(pop)

#define DEFALUT_INTENSITY 25.0F

class Device
{
private:
	static uint32_t constexpr MAX_TPB_500{ 1024u };
	static uint32_t constexpr MAX_TPB_350{ 384u };

public:
	int deviceID;
	std::string name;
	uint32_t computeVersion;
	float intensity;

	bool initialized;
	bool mining;

	std::thread miningThread;
	std::atomic<uint64_t> hashCount;
	std::atomic<std::chrono::steady_clock::time_point> hashStartTime;

	uint64_t* d_Solutions;
	uint64_t* h_Solutions;
	uint32_t* d_SolutionCount;
	uint32_t* h_SolutionCount;

private:
	dim3 m_block;
	dim3 m_grid;

	uint32_t m_lastCompute;
	uint32_t m_lastBlockX;
	uint32_t m_lastThreads;
	float m_lastIntensity;

public:
	static bool foundNvAPI64();

	Device();

	int CoreOC();
	int MemoryOC();

	uint32_t threads();
	dim3 block();
	dim3 grid();

	uint64_t hashRate();
};
