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
	typedef void(*MessageCallback)(int, const char*, const char*);
	typedef void(*SolutionCallback)(const char*, const char*, const char*, const char*, const char*, const char*, bool);

	bool isSubmitStale;

private:
	MessageCallback m_messageCallback;
	SolutionCallback m_solutionCallback;
	std::vector<std::unique_ptr<Device>> m_devices;

	static std::atomic<bool> m_newTarget;
	static std::atomic<bool> m_newMessage;

	std::string s_address;
	std::string s_challenge;
	std::string s_target;
	std::string s_difficulty;
	std::string s_customDifficulty;
	
	address_t m_address;
	byte32_t m_challenge;
	byte32_t m_solution;
	prefix_t m_prefix; // challenge32 + address20
	message_t m_miningMessage; // challenge32 + address20 + solution32

	arith_uint256 m_target;
	arith_uint256 m_difficulty;
	arith_uint256 m_maxDifficulty;
	arith_uint256 m_customDifficulty;

	std::atomic<uint64_t> m_solutionHashCount;
	std::atomic<std::chrono::steady_clock::time_point> m_solutionHashStartTime;

	std::mutex m_checkInputsMutex;
	std::mutex m_searchSpaceMutex;
	std::set<byte32_t> m_oldChallenges;
	std::thread m_runThread;

public:
	static std::string getCudaErrorString(cudaError_t &error);
	static int getDeviceCount(std::string &errorMessage);
	static std::string getDeviceName(int deviceID, std::string &errorMessage);
	
	// require web3 contract getMethod -> _MAXIMUM_TARGET
	CUDASolver(std::string const maxDifficulty) noexcept;
	~CUDASolver();

	void setMessageCallback(MessageCallback messageCallback);
	void setSolutionCallback(SolutionCallback solutionCallback);

	bool assignDevice(int const deviceID, float const intensity);
	bool isAssigned();
	bool isMining();

	void updatePrefix(std::string const prefix);
	void updateTarget(std::string const target);
	void updateDifficulty(std::string const difficulty);
	void setCustomDifficulty(uint32_t customDifficulty);

	void startFinding();
	void stopFinding();

	uint64_t getTotalHashRate();
	uint64_t getHashRateByDeviceID(int const deviceID);

private:
	void onMessage(int deviceID, const char* type, const char* message);
	void onMessage(int deviceID, std::string type, std::string message);

	const std::string keccak256(std::string const message); // for CPU verification
	void onSolution(byte32_t const solution);

	void findSolution(int const deviceID);
	void checkInputs(std::unique_ptr<Device>& device);
	void pushTarget();
	void pushMessage();
	void submitSolutions(std::set<uint64_t> solutions);

	uint64_t getNextSearchSpace(std::unique_ptr<Device>& device);
	const state_t getMidState(message_t &newMessage);
};
