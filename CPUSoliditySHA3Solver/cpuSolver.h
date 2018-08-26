#include <cassert>
#include <chrono>
#include <string>
#include <random>
#include <vector>
#include "types.h"
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

class cpuSolver
{
public:
	typedef void(*GetKingAddressCallback)(uint8_t *);
	typedef void(*GetWorkPositionCallback)(uint64_t &);
	typedef void(*ResetWorkPositionCallback)(uint64_t &);
	typedef void(*IncrementWorkPositionCallback)(uint64_t &, uint64_t);
	typedef void(*GetSolutionTemplateCallback)(uint8_t *);
	typedef void(*MessageCallback)(int, const char *, const char *);
	typedef void(*SolutionCallback)(const char *, const char *, const char *, const char *, const char *);

	bool m_SubmitStale;

private:
	static bool m_pause;

	GetKingAddressCallback m_getKingAddressCallback;
	GetSolutionTemplateCallback m_getSolutionTemplateCallback;
	GetWorkPositionCallback m_getWorkPositionCallback;
	ResetWorkPositionCallback m_resetWorkPositionCallback;
	IncrementWorkPositionCallback m_incrementWorkPositionCallback;
	MessageCallback m_messageCallback;
	SolutionCallback m_solutionCallback;

	byte32_t b_target;
	arith_uint256 m_target;

	std::string s_address;
	std::string s_challenge;
	std::string s_target;

	address_t m_address;
	address_t m_kingAddress;
	prefix_t m_prefix; // challenge32 + address20

	uint32_t m_miningThreadCount;
	uint32_t* m_miningThreadAffinities;
	bool* m_isThreadMining;

	uint64_t* m_threadHashes;
	std::chrono::steady_clock::time_point m_hashStartTime;
	std::chrono::steady_clock::time_point m_solutionHashStartTime;

public:
	static uint32_t getLogicalProcessorsCount();
	static std::string getNewSolutionTemplate(std::string kingAddress = "");

	cpuSolver(std::string const threads) noexcept;
	~cpuSolver() noexcept;

	void setGetKingAddressCallback(GetKingAddressCallback kingAddressCallback);
	void setGetWorkPositionCallback(GetWorkPositionCallback workPositionCallback);
	void setResetWorkPositionCallback(ResetWorkPositionCallback resetWorkPositionCallback);
	void setIncrementWorkPositionCallback(IncrementWorkPositionCallback incrementWorkPositionCallback);
	void setGetSolutionTemplateCallback(GetSolutionTemplateCallback solutionTemplateCallback);
	void setMessageCallback(MessageCallback messageCallback);
	void setSolutionCallback(SolutionCallback solutionCallback);

	bool isMining();
	bool isPaused();

	void updatePrefix(std::string const prefix);
	void updateTarget(std::string const target);

	uint64_t getTotalHashRate();
	uint64_t getHashRateByThreadID(uint32_t const threadID);

	void startFinding();
	void stopFinding();
	void pauseFinding(bool pauseFinding);

private:
	bool islessThan(byte32_t &left, byte32_t &right);
	bool isAddressEmpty(address_t kingAddress);
	void getKingAddress(address_t *kingAddress);
	void getSolutionTemplate(byte32_t *solutionTemplate);
	void getWorkPosition(uint64_t &workPosition);
	void resetWorkPosition(uint64_t &lastPosition);
	void incrementWorkPosition(uint64_t &lastPosition, uint64_t increment);
	void onMessage(int threadID, const char* type, const char* message);
	void onMessage(int threadID, std::string type, std::string message);
	void onSolution(byte32_t const solution, byte32_t const digest, std::string challenge);
	bool setCurrentThreadAffinity(uint32_t const affinityMask);
	void findSolution(uint32_t const threadID, uint32_t const affinityMask);
};
