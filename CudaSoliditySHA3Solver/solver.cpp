#include "solver.h"

int Solver::getDeviceCount(std::string &errorMessage) // static
{
	errorMessage = "";
	int deviceCount;
	cudaError_t response = cudaGetDeviceCount(&deviceCount);

	if (response == cudaError::cudaSuccess) return deviceCount;
	else
	{
		errorMessage = getCudaErrorString(response);
		return 0;
	}
}

std::string Solver::getDeviceName(int deviceID, std::string &errorMessage) // static
{
	errorMessage = "";
	cudaDeviceProp devProp;
	cudaError_t response = cudaGetDeviceProperties(&devProp, deviceID);

	if (response == cudaError::cudaSuccess) return std::string(devProp.name);
	else
	{
		errorMessage = getCudaErrorString(response);
		return "";
	}
}

std::string Solver::getCudaErrorString(cudaError_t err) // static
{
	return std::string(cudaGetErrorString(err));
}

Solver::Solver() :
	s_address{ "" },
	s_challenge{ "" },
	s_target{ "" },
	s_difficulty{ "" },
	m_address{ 0 },
	m_challenge{ 0 },
	m_target{ 0ull }
{
	device_t::target = 0ull;
	device_t::newTarget = false;
	device_t::newMessage = false;
	device_t::stop = false;

	reinterpret_cast<uint64_t&>(m_solution[0]) = 06055134500533075101ull;

	std::random_device r;
	std::mt19937_64 gen{ r() };
	std::uniform_int_distribution<uint64_t> urInt{ 0, UINT64_MAX };

	for (uint_fast8_t i_rand{ 8 }; i_rand < UINT256_LENGTH; i_rand += 8)
		reinterpret_cast<uint64_t&>(m_solution[i_rand]) = urInt(gen);

	std::memset(&m_solution[12], 0, 8);

	std::memcpy(&m_message[PREFIX_LENGTH], m_solution.data(), UINT256_LENGTH);
}

Solver::~Solver()
{
	// TODO

}

bool Solver::assignDevice(int const deviceID, float const intensity)
{
	onMessage(deviceID, "Info", "Assigning device...");

	struct cudaDeviceProp deviceProp;
	cudaError_t response = cudaGetDeviceProperties(&deviceProp, deviceID);
	if (response != cudaSuccess)
	{
		onMessage(deviceID, "Error", cudaGetErrorString(response));
		return false;
	}
	m_cudaSolvers.emplace_back(new CUDASolver(deviceID, deviceProp, intensity, this));
	return true;
}

void Solver::startSolving()
{
	while (!device_t::newMessage || !device_t::newTarget)
		std::this_thread::sleep_for(std::chrono::milliseconds(500));

	std::for_each(m_cudaSolvers.begin(), m_cudaSolvers.end(), [](CUDASolver * solver)
	{
		solver->startFinding();
	});
}

void Solver::onMessage(int deviceID, const char* type, const char* message)
{
	m_messageCallback(deviceID, type, message);
}

void Solver::onMessage(int deviceID, std::string type, std::string message)
{
	onMessage(deviceID, type.c_str(), message.c_str());
}

void Solver::onSolution(int deviceID, hash_t solution)
{
	std::string solutionStr{ bytesToHexString(solution) };

	onMessage(deviceID, "Info", "Submitting solution...");
	m_solutionCallback(deviceID, s_address.c_str(), s_challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), solutionStr.c_str());
}

const std::string Solver::keccak256(std::string const message)
{
	message_t data;
	hexStringToBytes(message, data);

	sph_keccak256_context ctx;
	sph_keccak256_init(&ctx);
	sph_keccak256(&ctx, data.data(), data.size());

	hash_t out;
	sph_keccak256_close(&ctx, out.data());

	return bytesToHexString(out);
}
