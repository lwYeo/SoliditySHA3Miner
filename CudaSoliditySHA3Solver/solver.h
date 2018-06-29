#pragma once

#include <random>
#include "cudaSolver.h"
#include "types.h"
#include "UINT256/arith_uint256.h"

#define SPH_KECCAK_64 1
#include "sph_keccak.h"

struct device_t;
class CUDASolver;

class Solver
{
public:
	typedef void(*MessageCallback)(int, const char*, const char*);
	typedef void(*SolutionCallback)(int, const char*, const char*, const char*, const char*, const char*);

	std::string s_address;
	std::string s_challenge;
	std::string s_target;
	std::string s_difficulty;
	hash_t m_solution;
	message_t m_message;
	hash_t m_address;
	hash_t m_challenge;
	uint64_t m_target;

private:
	std::vector<CUDASolver *> m_cudaSolvers;
	MessageCallback m_messageCallback;
	SolutionCallback m_solutionCallback;

public:
	static int getDeviceCount(std::string &errorMessage);
	static std::string getDeviceName(int deviceID, std::string &errorMessage);

private:
	static std::string getCudaErrorString(cudaError_t err);

public:
	Solver();
	~Solver();

	void onMessage(int deviceID, const char* type, const char* message);
	void onMessage(int deviceID, std::string type, std::string message);
	void onSolution(int deviceID, hash_t solution);

	bool assignDevice(int const deviceID, float const intensity);
	void startSolving();
	const std::string keccak256(std::string const message); // for CPU verification


private:

	
};