#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <thread>
#include "nv_api.h"
#include "../types.h"

namespace CUDASolver
{
	constexpr float DEFALUT_INTENSITY{ 25.0f };

	class Device
	{
	public:
		int deviceID;
		std::string name;
		uint32_t computeVersion;
		float intensity;

		bool initialized;
		bool mining;

		std::thread miningThread;
		std::atomic<uint64_t> hashCount;
		std::chrono::steady_clock::time_point hashStartTime;

		uint64_t* d_Solutions;
		uint64_t* h_Solutions;
		uint32_t* d_SolutionCount;
		uint32_t* h_SolutionCount;

		bool checkChanges;
		bool isNewTarget;
		bool isNewMessage;

		message_ut currentMessage;
		sponge_ut currentMidstate;
		byte32_t currentTarget;
		uint64_t currentHigh64Target;

	private:
		dim3 m_block;
		dim3 m_grid;

		uint32_t m_lastCompute;
		uint32_t m_lastBlockX;
		uint32_t m_lastThreads;
		float m_lastIntensity;

		NV_API m_api;
		uint32_t pciBusID;

	public:
		Device(int deviceID);

		bool getSettingMaxCoreClock(int *maxCoreClock, std::string *errorMessage);
		bool getSettingMaxMemoryClock(int *maxMemoryClock, std::string *errorMessage);
		bool getSettingPowerLimit(int *powerLimit, std::string *errorMessage);
		bool getSettingThermalLimit(int *thermalLimit, std::string *errorMessage);
		bool getSettingFanLevelPercent(int *fanLevel, std::string *errorMessage);

		bool getCurrentFanTachometerRPM(int *tachometerRPM, std::string *errorMessage);
		bool getCurrentTemperature(int *temperature, std::string *errorMessage);
		bool getCurrentCoreClock(int *coreClock, std::string *errorMessage);
		bool getCurrentMemoryClock(int *memoryClock, std::string *errorMessage);
		bool getCurrentUtilizationPercent(int *utilization, std::string *errorMessage);
		bool getCurrentPstate(int *pstate, std::string *errorMessage);
		bool getCurrentThrottleReasons(std::string *reasons, std::string *errorMessage);

		uint32_t threads();
		dim3 block();
		dim3 grid();

		uint64_t hashRate();
	};
}
