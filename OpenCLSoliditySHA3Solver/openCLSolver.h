#pragma once

#define MAX_WORK_POSITION_STORE 2

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include "sha3.h"
#include "device/device.h"
#include "uint256/arith_uint256.h"

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

namespace OpenCLSolver
{
	typedef void(*GetKingAddressCallback)(uint8_t *kingAddress);
	typedef void(*GetSolutionTemplateCallback)(uint8_t *solutionTemplate);
	typedef void(*GetWorkPositionCallback)(uint64_t &lastWorkPosition);
	typedef void(*ResetWorkPositionCallback)(uint64_t &lastWorkPosition);
	typedef void(*IncrementWorkPositionCallback)(uint64_t &lastWorkPosition, uint64_t incrementSize);
	typedef void(*MessageCallback)(const char *platform, int deviceEnum, const char *type, const char *message);
	typedef void(*SolutionCallback)(const char *digest, const char *address, const char *challenge, const char *target, const char *solution);

	typedef struct { cl_platform_id id; std::string name; } Platform;

	class openCLSolver
	{
	public:
		static bool foundAdlApi();
		static void preInitialize(bool allowIntel, std::string &errorMessage);
		static std::string getPlatformNames();
		static int getDeviceCount(std::string platformName, std::string &errorMessage);
		static std::string getDeviceName(std::string platformName, int deviceEnum, std::string &errorMessage);

		GetKingAddressCallback m_getKingAddressCallback;
		GetSolutionTemplateCallback m_getSolutionTemplateCallback;
		GetWorkPositionCallback m_getWorkPositionCallback;
		ResetWorkPositionCallback m_resetWorkPositionCallback;
		IncrementWorkPositionCallback m_incrementWorkPositionCallback;
		MessageCallback m_messageCallback;
		SolutionCallback m_solutionCallback;

		bool isSubmitStale;

	private:
		static std::vector<Platform> platforms;

		std::vector<std::unique_ptr<Device>> m_devices;
		std::thread m_runThread;

		static bool m_pause;
		static bool m_isSubmitting;
		static bool m_isKingMaking;

		std::string s_address;
		std::string s_challenge;
		std::string s_target;

		address_t m_address;
		address_t m_kingAddress;
		byte32_t m_solutionTemplate;
		message_ut m_miningMessage;
		arith_uint256 m_target;

	public:
		// require web3 contract getMethod -> _MAXIMUM_TARGET
		openCLSolver() noexcept;
		~openCLSolver() noexcept;

		void setGetKingAddressCallback(GetKingAddressCallback kingAddressCallback);
		void setGetSolutionTemplateCallback(GetSolutionTemplateCallback solutionTemplateCallback);
		void setGetWorkPositionCallback(GetWorkPositionCallback workPositionCallback);
		void setResetWorkPositionCallback(ResetWorkPositionCallback resetWorkPositionCallback);
		void setIncrementWorkPositionCallback(IncrementWorkPositionCallback incrementWorkPositionCallback);
		void setMessageCallback(MessageCallback messageCallback);
		void setSolutionCallback(SolutionCallback solutionCallback);

		bool isAssigned();
		bool isAnyInitialised();
		bool isMining();
		bool isPaused();
		bool assignDevice(std::string platformName, int deviceEnum, float &intensity, unsigned int &pciBusID, const char *deviceName, uint64_t *nameSize);

		void updatePrefix(std::string const prefix);
		void updateTarget(std::string const target);

		uint64_t getTotalHashRate();
		uint64_t getHashRateByDevice(std::string platformName, int const deviceEnum);

		int getDeviceSettingMaxCoreClock(std::string platformName, int deviceEnum);
		int getDeviceSettingMaxMemoryClock(std::string platformName, int deviceEnum);
		int getDeviceSettingPowerLimit(std::string platformName, int deviceEnum);
		int getDeviceSettingThermalLimit(std::string platformName, int deviceEnum);
		int getDeviceSettingFanLevelPercent(std::string platformName, int deviceEnum);

		int getDeviceCurrentFanTachometerRPM(std::string platformName, int deviceEnum);
		int getDeviceCurrentTemperature(std::string platformName, int deviceEnum);
		int getDeviceCurrentCoreClock(std::string platformName, int deviceEnum);
		int getDeviceCurrentMemoryClock(std::string platformName, int deviceEnum);
		int getDeviceCurrentUtilizationPercent(std::string platformName, int deviceEnum);

		void startFinding();
		void stopFinding();
		void pauseFinding(bool pauseFinding);

	private:
		bool isAddressEmpty(address_t &address);
		void getKingAddress(address_t *kingAddress);
		void getSolutionTemplate(byte32_t *solutionTemplate);
		void getWorkPosition(uint64_t &workPosition);
		void resetWorkPosition(uint64_t &lastPosition);
		void incrementWorkPosition(uint64_t &lastPosition, uint64_t increment);
		void onMessage(std::string platformName, int deviceEnum, std::string type, std::string message);
		void onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device);

		void findSolution(std::string platformName, int const deviceEnum);
		void checkInputs(std::unique_ptr<Device> &device, char *currentChallenge);
		void pushTarget(std::unique_ptr<Device> &device);
		void pushTargetKing(std::unique_ptr<Device> &device);
		void pushMessage(std::unique_ptr<Device> &device);
		void pushMessageKing(std::unique_ptr<Device> &device);
		void submitSolutions(std::set<uint64_t> solutions, std::string challenge, std::string platformName, int const deviceEnum);

		uint64_t const getNextWorkPosition(std::unique_ptr<Device> &device);
		sponge_ut const getMidState(message_ut &newMessage);
	};
}
