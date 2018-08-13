#pragma once

#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

#define MAX_WORK_POSITION_STORE 4

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <set>
#include "device/device.h"
#include "uint256/arith_uint256.h"

#define SPH_KECCAK_64 1
#include "sph/sph_keccak.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <thread>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

#	define _M_CEE 001 
#else 
#	include <thread>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

#endif 

#pragma managed(pop)

class openCLSolver
{
public:
	typedef void(*GetSolutionTemplateCallback)(uint8_t *&);
	typedef void(*GetWorkPositionCallback)(uint64_t &);
	typedef void(*ResetWorkPositionCallback)(uint64_t &);
	typedef void(*IncrementWorkPositionCallback)(uint64_t &, uint64_t);
	typedef void(*MessageCallback)(const char *, int, const char *, const char *);
	typedef void(*SolutionCallback)(const char *, const char *, const char *, const char *, const char *, const char *, bool);
	typedef struct { cl_platform_id id; std::string name; } Platform;

	static void preInitialize(bool allowIntel, std::string &errorMessage);
	static std::string getPlatformNames();
	static int getDeviceCount(std::string platformName, std::string &errorMessage);
	static std::string getDeviceName(std::string platformName, int deviceEnum, std::string &errorMessage);

	bool isSubmitStale;

private:
	static std::vector<Platform> platforms;

	GetSolutionTemplateCallback m_getSolutionTemplateCallback;
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
	uint8_t *m_solutionTemplate;
	prefix_t m_prefix; // challenge32 + address20
	message_t m_miningMessage; // challenge32 + address20 + solution32

	arith_uint256 m_target;
	arith_uint256 m_difficulty;
	arith_uint256 m_maxDifficulty;
	arith_uint256 m_customDifficulty;

	std::atomic<std::chrono::steady_clock::time_point> m_solutionHashStartTime;
	std::thread m_runThread;

public:
	// require web3 contract getMethod -> _MAXIMUM_TARGET
	openCLSolver(std::string const maxDifficulty, std::string kingAddress) noexcept;
	~openCLSolver() noexcept;

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
	bool assignDevice(std::string platformName, int deviceEnum, float const intensity);

	void updatePrefix(std::string const prefix);
	void updateTarget(std::string const target);
	void updateDifficulty(std::string const difficulty);
	void setCustomDifficulty(uint32_t customDifficulty);

	uint64_t getTotalHashRate();
	uint64_t getHashRateByDevice(std::string platformName, int const deviceEnum);

	void startFinding();
	void stopFinding();
	void pauseFinding(bool pauseFinding);

private:
	void getSolutionTemplate(uint8_t *&solutionTemplate);
	void getWorkPosition(uint64_t &workPosition);
	void resetWorkPosition(uint64_t &lastPosition);
	void incrementWorkPosition(uint64_t &lastPosition, uint64_t increment);
	void onMessage(std::string platformName, int deviceEnum, std::string type, std::string message);
	void onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device);

	const std::string keccak256(std::string const message); // for CPU verification

	void findSolution(std::string platformName, int const deviceEnum);
	void setKernelArgs(std::unique_ptr<Device> &device);
	void checkInputs(std::unique_ptr<Device> &device, char *currentChallenge);
	void pushTarget(std::unique_ptr<Device> &device);
	void pushMessage(std::unique_ptr<Device> &device);
	void submitSolutions(std::set<uint64_t> solutions, std::string challenge, std::string platformName, int const deviceEnum);

	uint64_t const getNextWorkPosition(std::unique_ptr<Device> &device);
	state_t const getMidState(message_t &newMessage);
};
