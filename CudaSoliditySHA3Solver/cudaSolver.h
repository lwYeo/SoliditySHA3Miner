#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <set>
#include "sha3.h"
#include "device\device.h"
#include "uint256/arith_uint256.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <thread> 
#	define _M_CEE 001 
#else 
#	include <thread> 
#endif 

#pragma managed(pop)

// --------------------------------------------------------------------
// CUDA common constants
// --------------------------------------------------------------------

#ifdef __INTELLISENSE__
//	reduce vstudio warnings (__byteperm, blockIdx...)
#	include <device_functions.h>
#	include <device_launch_parameters.h>
#endif //__INTELLISENSE__

#define MAX_SOLUTION_COUNT_DEVICE			32
#define NONCE_POSITION						UINT256_LENGTH + ADDRESS_LENGTH + ADDRESS_LENGTH
#define KING_STRIDE							8

__constant__ static uint64_t const Keccak_f1600_RC[24] =
{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

class CUDASolver
{
public:
	typedef void(*GetKingAddressCallback)(uint8_t *);
	typedef void(*GetSolutionTemplateCallback)(uint8_t *);
	typedef void(*GetWorkPositionCallback)(uint64_t &);
	typedef void(*ResetWorkPositionCallback)(uint64_t &);
	typedef void(*IncrementWorkPositionCallback)(uint64_t &, uint64_t);
	typedef void(*MessageCallback)(int, const char *, const char *);
	typedef void(*SolutionCallback)(const char *, const char *, const char *, const char *, const char *);

	bool isSubmitStale;

private:
	GetKingAddressCallback m_getKingAddressCallback;
	GetSolutionTemplateCallback m_getSolutionTemplateCallback;
	GetWorkPositionCallback m_getWorkPositionCallback;
	ResetWorkPositionCallback m_resetWorkPositionCallback;
	IncrementWorkPositionCallback m_incrementWorkPositionCallback;
	MessageCallback m_messageCallback;
	SolutionCallback m_solutionCallback;
	std::vector<std::unique_ptr<Device>> m_devices;

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

	std::atomic<std::chrono::steady_clock::time_point> m_solutionHashStartTime;
	std::thread m_runThread;

public:
	static bool foundNvAPI64();

	static std::string getCudaErrorString(cudaError_t &error);
	static int getDeviceCount(std::string &errorMessage);
	static std::string getDeviceName(int deviceID, std::string &errorMessage);
	
	// require web3 contract getMethod -> _MAXIMUM_TARGET
	CUDASolver() noexcept;
	~CUDASolver() noexcept;

	void setGetKingAddressCallback(GetKingAddressCallback kingAddressCallback);
	void setGetSolutionTemplateCallback(GetSolutionTemplateCallback solutionTemplateCallback);
	void setGetWorkPositionCallback(GetWorkPositionCallback workPositionCallback);
	void setResetWorkPositionCallback(ResetWorkPositionCallback resetWorkPositionCallback);
	void setIncrementWorkPositionCallback(IncrementWorkPositionCallback incrementWorkPositionCallback);
	void setMessageCallback(MessageCallback messageCallback);
	void setSolutionCallback(SolutionCallback solutionCallback);

	bool assignDevice(int const deviceID, float const intensity);
	bool isAssigned();
	bool isAnyInitialised();
	bool isMining();
	bool isPaused();

	void updatePrefix(std::string const prefix);
	void updateTarget(std::string const target);

	void startFinding();
	void stopFinding();
	void pauseFinding(bool pauseFinding);

	uint64_t getTotalHashRate();
	uint64_t getHashRateByDeviceID(int const deviceID);

	int getDeviceSettingMaxCoreClock(int deviceID);
	int getDeviceSettingMaxMemoryClock(int deviceID);
	int getDeviceSettingPowerLimit(int deviceID);
	int getDeviceSettingThermalLimit(int deviceID);
	int getDeviceSettingFanLevelPercent(int deviceID);

	int getDeviceCurrentFanTachometerRPM(int deviceID);
	int getDeviceCurrentTemperature(int deviceID);
	int getDeviceCurrentCoreClock(int deviceID);
	int getDeviceCurrentMemoryClock(int deviceID);
	int getDeviceCurrentUtilizationPercent(int deviceID);
	int getDeviceCurrentPstate(int deviceID);
	std::string getDeviceCurrentThrottleReasons(int deviceID);

private:
	void initializeDevice(std::unique_ptr<Device> &device);
	bool isAddressEmpty(address_t &address);
	void getKingAddress(address_t *kingAddress);
	void getSolutionTemplate(byte32_t *solutionTemplate);
	void getWorkPosition(uint64_t &workPosition);
	void resetWorkPosition(uint64_t &lastPosition);
	void incrementWorkPosition(uint64_t &lastPosition, uint64_t increment);
	void onMessage(int deviceID, const char *type, const char *message);
	void onMessage(int deviceID, std::string type, std::string message);
	
	void onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device);

	void findSolution(int const deviceID);
	void findSolutionKing(int const deviceID);
	void checkInputs(std::unique_ptr<Device> &device, char *currentChallenge);
	void pushTarget(std::unique_ptr<Device> &device);
	void pushTargetKing(std::unique_ptr<Device> &device);
	void pushMessage(std::unique_ptr<Device> &device);
	void pushMessageKing(std::unique_ptr<Device> &device);
	void submitSolutions(std::set<uint64_t> solutions, std::string challenge, int const deviceID);

	uint64_t getNextWorkPosition(std::unique_ptr<Device>& device);
	sponge_ut const getMidState(message_ut &newMessage);
};
