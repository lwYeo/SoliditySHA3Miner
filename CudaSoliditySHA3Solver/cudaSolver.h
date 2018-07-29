#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <set>
#include "device\device.h"
#include "uint256/arith_uint256.h"

#define SPH_KECCAK_64 1
#include "sph\sph_keccak.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <thread> 
#	define _M_CEE 001 
#else 
#	include <thread> 
#endif 

#pragma managed(pop)

#ifdef __INTELLISENSE__
// reduce vstudio warnings (__byteperm, blockIdx...)
#include <device_functions.h>
#include <device_launch_parameters.h>
#endif //__INTELLISENSE__

class CUDASolver
{
public:
	typedef void(*GetWorkPositionCallback)(uint64_t &);
	typedef void(*ResetWorkPositionCallback)(uint64_t &);
	typedef void(*IncrementWorkPositionCallback)(uint64_t &, uint64_t);
	typedef void(*MessageCallback)(int, const char*, const char*);
	typedef void(*SolutionCallback)(const char*, const char*, const char*, const char*, const char*, const char*, bool);

	bool isSubmitStale;

private:
	GetWorkPositionCallback m_getWorkPositionCallback;
	ResetWorkPositionCallback m_resetWorkPositionCallback;
	IncrementWorkPositionCallback m_incrementWorkPositionCallback;
	MessageCallback m_messageCallback;
	SolutionCallback m_solutionCallback;
	std::vector<std::unique_ptr<Device>> m_devices;

	static bool m_pause;

	std::string s_kingAddress;
	std::string s_address;
	std::string s_challenge;
	std::string s_target;
	std::string s_difficulty;
	std::string s_customDifficulty;
	
	address_t m_address;
	byte32_t m_solutionTemplate;
	prefix_t m_prefix; // challenge32 + address20
	message_t m_miningMessage; // challenge32 + address20 + solution32

	arith_uint256 m_target;
	arith_uint256 m_difficulty;
	arith_uint256 m_maxDifficulty;
	arith_uint256 m_customDifficulty;

	std::atomic<std::chrono::steady_clock::time_point> m_solutionHashStartTime;
	std::thread m_runThread;

public:
	static std::string getCudaErrorString(cudaError_t &error);
	static int getDeviceCount(std::string &errorMessage);
	static std::string getDeviceName(int deviceID, std::string &errorMessage);
	
	// require web3 contract getMethod -> _MAXIMUM_TARGET
	CUDASolver(std::string const maxDifficulty, std::string solutionTemplate, std::string kingAddress) noexcept;
	~CUDASolver() noexcept;

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
	void updateDifficulty(std::string const difficulty);
	void setCustomDifficulty(uint32_t const customDifficulty);

	void startFinding();
	void stopFinding();
	void pauseFinding(bool pauseFinding);

	uint64_t getTotalHashRate();
	uint64_t getHashRateByDeviceID(int const deviceID);

private:
	void getWorkPosition(uint64_t &workPosition);
	void resetWorkPosition(uint64_t &lastPosition);
	void incrementWorkPosition(uint64_t &lastPosition, uint64_t increment);
	void onMessage(int deviceID, const char *type, const char *message);
	void onMessage(int deviceID, std::string type, std::string message);
	
	// for CPU verification
	const std::string keccak256(std::string const message);
	void onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device);

	void findSolution(int const deviceID);
	void checkInputs(std::unique_ptr<Device> &device, char *currentChallenge);
	void pushTarget(std::unique_ptr<Device> &device);
	void pushMessage(std::unique_ptr<Device> &device);
	void submitSolutions(std::set<uint64_t> solutions, std::string challenge, int const deviceID);

	uint64_t getNextWorkPosition(std::unique_ptr<Device>& device);
	state_t const getMidState(message_t &newMessage);
};
