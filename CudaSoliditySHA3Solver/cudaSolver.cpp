#pragma unmanaged

#include "cudasolver.h"

#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

std::atomic<bool> CUDASolver::m_newTarget;
std::atomic<bool> CUDASolver::m_newMessage;

std::string CUDASolver::getCudaErrorString(cudaError_t &error)
{
	return std::string(cudaGetErrorString(error));
}

int CUDASolver::getDeviceCount(std::string &errorMessage)
{
	errorMessage = "";
	int deviceCount;
	cudaError_t response = cudaGetDeviceCount(&deviceCount);

	if (response == cudaError::cudaSuccess) return deviceCount;
	else errorMessage = getCudaErrorString(response);

	return 0;
}

std::string CUDASolver::getDeviceName(int deviceID, std::string &errorMessage)
{
	errorMessage = "";
	cudaDeviceProp devProp;
	cudaError_t response = cudaGetDeviceProperties(&devProp, deviceID);

	if (response == cudaError::cudaSuccess) return std::string(devProp.name);
	else errorMessage = getCudaErrorString(response);

	return "";
}

// --------------------------------------------------------------------
// Public
// --------------------------------------------------------------------

CUDASolver::CUDASolver(std::string const maxDifficulty) noexcept :
	s_address{ "" },
	s_challenge{ "" },
	s_target{ "" },
	s_difficulty{ "" },
	m_address{ 0 },
	m_challenge{ 0 },
	m_solution{ 0 },
	m_prefix{ 0 },
	m_miningMessage{ 0 },
	m_target{ 0 },
	m_difficulty{ 0 },
	m_maxDifficulty{ maxDifficulty }
{
	m_newTarget.store(false);
	m_newMessage.store(false);

	m_solutionHashCount.store(0);
	m_solutionHashStartTime.store(std::chrono::steady_clock::now());

	{
		std::random_device rand;
		std::mt19937_64 rGen{ rand() };
		std::uniform_int_distribution<uint64_t> uInt_d{ 0, UINT64_MAX };

		reinterpret_cast<uint64_t&>(m_solution[0]) = 06055134500533075101ull;

		for (uint_fast8_t i{ UINT64_LENGTH }; i < UINT256_LENGTH; i += UINT64_LENGTH)
			reinterpret_cast<uint64_t&>(m_solution[i]) = uInt_d(rGen);

		std::memset(&m_solution[12], 0, UINT64_LENGTH); // keep first and last 12 bytes, leave middle 8 bytes for mid state
	}
}

CUDASolver::~CUDASolver() noexcept
{
	stopFinding();
}

void CUDASolver::setMessageCallback(MessageCallback messageCallback)
{
	m_messageCallback = messageCallback;
}

void CUDASolver::setSolutionCallback(SolutionCallback solutionCallback)
{
	m_solutionCallback = solutionCallback;
}

bool CUDASolver::assignDevice(int const deviceID, float const intensity)
{
	onMessage(deviceID, "Info", "Assigning device...");

	struct cudaDeviceProp deviceProp;
	cudaError_t response = cudaGetDeviceProperties(&deviceProp, deviceID);
	if (response != cudaSuccess)
	{
		onMessage(deviceID, "Error", cudaGetErrorString(response));
		return false;
	}

	m_devices.push_back(std::make_unique<Device>());
	auto& assignDevice = m_devices.back();

	assignDevice->deviceID = deviceID;
	assignDevice->name = deviceProp.name;
	assignDevice->computeVersion = deviceProp.major * 100 + deviceProp.minor * 10;

	std::string deviceName{ assignDevice->name };
	std::transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::toupper);

	float defaultIntensity{ DEFALUT_INTENSITY };

	if (deviceName.find("1180") != std::string::npos || deviceName.find("1080") != std::string::npos
		|| deviceName.find("1070 TI") != std::string::npos || deviceName.find("1070TI") != std::string::npos)
		defaultIntensity = 29.0F;
	else if (deviceName.find("1070") != std::string::npos || deviceName.find("980") != std::string::npos)
		defaultIntensity = 28.0F;
	else if (deviceName.find("1060") != std::string::npos || deviceName.find("1050") != std::string::npos
		|| deviceName.find("970") != std::string::npos)
		defaultIntensity = 27.0F;

	assignDevice->intensity = (intensity < 1.000F) ? defaultIntensity : intensity;

	std::string message = "Assigned CUDA device (" + assignDevice->name + ")...";
#ifndef NDEBUG
	message += "\n Compute capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
#endif // !NDEBUG
	message += "\n Intensity: " + std::to_string(assignDevice->intensity);

	if (!assignDevice->foundNvAPI64()) message += "NvAPI library not found.";
	else
	{
		message += "\n Core OC: " + std::to_string(assignDevice->CoreOC()) + "MHz";
		//if (assignDevice->CoreOC() <= 0)
			//message += " (Recommended to OC for improved performance)";

		message += "\n Memory OC: " + std::to_string(assignDevice->MemoryOC()) + "MHz";
		if (assignDevice->MemoryOC() > 0)
			message += " (Memory OC has no improvement to performance)";
	}

	onMessage(assignDevice->deviceID, "Info", message);

	return true;
}

bool CUDASolver::isAssigned()
{
	for (auto& device : m_devices)
		if (device->deviceID > -1)
			return true;

	return false;
}

bool CUDASolver::isAnyInitialised()
{
	for (auto& device : m_devices)
		if (device->deviceID > -1 && device->initialized)
			return true;

	return false;
}

bool CUDASolver::isMining()
{
	for (auto& device : m_devices)
		if (device->mining)
			return true;

	return false;
}

void CUDASolver::updatePrefix(std::string const prefix)
{
	assert(prefix.length() == (PREFIX_LENGTH * 2 + 2));

	prefix_t tempPrefix;
	hexStringToBytes(prefix, tempPrefix);

	if (tempPrefix == m_prefix) return;

	s_challenge = prefix.substr(0, 2 + UINT256_LENGTH * 2);
	s_address = "0x" + prefix.substr(2 + UINT256_LENGTH * 2, ADDRESS_LENGTH * 2);

	byte32_t oldChallenge;
	std::memcpy(&oldChallenge, &m_prefix, UINT256_LENGTH);

	if ((m_oldChallenges.find(oldChallenge) != m_oldChallenges.end()) || m_oldChallenges.size() == 0)
		m_oldChallenges.emplace(oldChallenge);

	while (m_oldChallenges.size() > 100)
		m_oldChallenges.erase(m_oldChallenges.begin());

	std::memcpy(&m_prefix, &tempPrefix, PREFIX_LENGTH);

	m_newMessage.store(true);
	onMessage(-1, "Info", "New challenge detected " + s_challenge.substr(0, 18) + "...");
}

void CUDASolver::updateTarget(std::string const target)
{
	if (m_customDifficulty > 0u && !(target == (m_maxDifficulty / m_customDifficulty).GetHex())) return;

	arith_uint256 tempTarget = arith_uint256(target);
	if (tempTarget == m_target) return;

	s_target = (target.substr(0, 2) == "0x") ? target : "0x" + target;
	m_target = tempTarget;

	m_newTarget.store(true);
	onMessage(-1, "Info", "New target detected " + s_target.substr(0, 18) + "...");
}

void CUDASolver::updateDifficulty(std::string const difficulty)
{
	if (m_customDifficulty > 0u) return;

	arith_uint256 oldDifficulity{ m_difficulty };
	s_difficulty = difficulty;
	m_difficulty = arith_uint256(difficulty);

	if ((m_maxDifficulty != 0ull) && (m_difficulty != oldDifficulity))
	{
		onMessage(-1, "Info", "New difficulity detected (" + std::to_string(m_difficulty.GetLow64()) + ")...");
		updateTarget((m_maxDifficulty / m_difficulty).GetHex());
	}
}

void CUDASolver::setCustomDifficulty(uint32_t customDifficulty)
{
	if (customDifficulty == 0u) return;

	s_customDifficulty = std::to_string(customDifficulty);
	m_customDifficulty = arith_uint256{ customDifficulty };

	onMessage(-1, "Info", "Custom difficulty (" + s_customDifficulty + ") detected...");
	updateTarget((m_maxDifficulty / m_customDifficulty).GetHex());
}

void CUDASolver::startFinding()
{
	using namespace std::chrono_literals;

	m_solutionHashCount.store(0ull);
	m_solutionHashStartTime.store(std::chrono::steady_clock::now());

	for (auto& device : m_devices)
	{
		device->miningThread = std::thread(&CUDASolver::findSolution, this, device->deviceID);
		std::this_thread::sleep_for(100ms);
	}
}

void CUDASolver::stopFinding()
{
	using namespace std::chrono_literals;

	for (auto& device : m_devices) device->mining = false;

	for (auto& device : m_devices)
	{
		while (!device->miningThread.joinable()) std::this_thread::sleep_for(100ms);
		device->miningThread.join();
	}

	m_newMessage.store(false);
	m_newTarget.store(false);
	try { if (m_oldChallenges.size() > 0) m_oldChallenges.clear(); }
	catch (const std::exception&) {}
}

uint64_t CUDASolver::getTotalHashRate()
{
	uint64_t totalHashRate{ 0ull };
	for (auto& device : m_devices)
		totalHashRate += device->hashRate();

	return totalHashRate;
}

uint64_t CUDASolver::getHashRateByDeviceID(int const deviceID)
{
	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			return device->hashRate();

	return 0ull;
}

// --------------------------------------------------------------------
// Private
// --------------------------------------------------------------------

void CUDASolver::onMessage(int deviceID, const char* type, const char* message)
{
	m_messageCallback(deviceID, type, message);
}

void CUDASolver::onMessage(int deviceID, std::string type, std::string message)
{
	onMessage(deviceID, type.c_str(), message.c_str());
}

const std::string CUDASolver::keccak256(std::string const message)
{
	message_t data;
	hexStringToBytes(message, data);

	sph_keccak256_context ctx;
	sph_keccak256_init(&ctx);
	sph_keccak256(&ctx, data.data(), data.size());

	byte32_t out;
	sph_keccak256_close(&ctx, out.data());

	return bytesToHexString(out);
}

void CUDASolver::onSolution(byte32_t const solution)
{
	prefix_t prefix;
	std::memcpy(&prefix, &m_miningMessage, PREFIX_LENGTH);
	
	onMessage(-1, "Info", "Found solution, verifying...");

	byte32_t emptySolution;
	std::memset(&emptySolution, 0u, UINT256_LENGTH);
	if (solution == emptySolution)
	{
		onMessage(-1, "Error", "Verification failed: empty solution"
			+ std::string("\nChallenge: ") + s_challenge
			+ "\nAddress: " + s_address
			+ "\nTarget: " + s_target);
		return;
	}

	std::string prefixStr{ bytesToHexString(prefix) };
	std::string solutionStr{ bytesToHexString(solution) };

	std::string digestStr = keccak256(prefixStr + solutionStr);
	arith_uint256 digest = arith_uint256(digestStr);
	onMessage(-1, "Debug", "Digest: 0x" + digestStr);

	if (digest >= m_target)
	{
		for (byte32_t oldChallenge : m_oldChallenges)
		{
			std::string oldDigest = keccak256(bytesToHexString(oldChallenge) + bytesToHexString(m_address) + solutionStr);
			arith_uint256 oldDigest_b = arith_uint256(oldDigest);
			if (oldDigest_b < m_target)
			{
				if (isSubmitStale)
				{
					onMessage(-1, "Warn", "Verified stale solution, submitting"
						+ std::string("\nChallenge: ") + s_challenge
						+ "\nAddress: " + s_address
						+ "\nSolution: 0x" + solutionStr
						+ "\nDigest: 0x" + digestStr
						+ "\nTarget: " + s_target);
					m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), s_challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str(), m_customDifficulty > 0u);
				}
				else
				{
					onMessage(-1, "Error", "Verification failed: stale solution"
						+ std::string("\nChallenge: ") + s_challenge
						+ "\nAddress: " + s_address
						+ "\nSolution: 0x" + solutionStr
						+ "\nDigest: 0x" + digestStr
						+ "\nTarget: " + s_target);
				}
				return;
			}
		}
		onMessage(-1, "Error", "Verification failed: invalid solution"
			+ std::string("\nChallenge: ") + s_challenge
			+ "\nAddress: " + s_address
			+ "\nSolution: 0x" + solutionStr
			+ "\nDigest: 0x" + digestStr
			+ "\nTarget: " + s_target);
	}
	else
	{
		onMessage(-1, "Info", "Solution verified, submitting nonce 0x" + solutionStr + "...");
		m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), s_challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str(), m_customDifficulty > 0u);
	}
}

void CUDASolver::submitSolutions(std::set<uint64_t> solutions)
{
	for (uint64_t midStateSolution : solutions)
	{
		byte32_t solution{ m_solution };
		std::memcpy(&solution[12], &midStateSolution, UINT64_LENGTH);

		std::thread t{ &CUDASolver::onSolution, this, solution };
		t.join();
	}
}

uint64_t CUDASolver::getNextSearchSpace(std::unique_ptr<Device>& device)
{
	std::lock_guard<std::mutex> lock(m_searchSpaceMutex);

	device->hashCount += device->threads();
	return m_solutionHashCount.fetch_add(device->threads());
}

const state_t CUDASolver::getMidState(message_t &newMessage)
{
	uint64_t message[11]{ 0 };
	std::memcpy(&message, &newMessage, MESSAGE_LENGTH);

	uint64_t C[5], D[5], mid[MIDSTATE_LENGTH];
	C[0] = message[0] ^ message[5] ^ message[10] ^ 0x100000000ull;
	C[1] = message[1] ^ message[6] ^ 0x8000000000000000ull;
	C[2] = message[2] ^ message[7];
	C[3] = message[3] ^ message[8];
	C[4] = message[4] ^ message[9];

	D[0] = ROTL64(C[1], 1) ^ C[4];
	D[1] = ROTL64(C[2], 1) ^ C[0];
	D[2] = ROTL64(C[3], 1) ^ C[1];
	D[3] = ROTL64(C[4], 1) ^ C[2];
	D[4] = ROTL64(C[0], 1) ^ C[3];

	mid[0] = message[0] ^ D[0];
	mid[1] = ROTL64(message[6] ^ D[1], 44);
	mid[2] = ROTL64(D[2], 43);
	mid[3] = ROTL64(D[3], 21);
	mid[4] = ROTL64(D[4], 14);
	mid[5] = ROTL64(message[3] ^ D[3], 28);
	mid[6] = ROTL64(message[9] ^ D[4], 20);
	mid[7] = ROTL64(message[10] ^ D[0] ^ 0x100000000ull, 3);
	mid[8] = ROTL64(0x8000000000000000ull ^ D[1], 45);
	mid[9] = ROTL64(D[2], 61);
	mid[10] = ROTL64(message[1] ^ D[1], 1);
	mid[11] = ROTL64(message[7] ^ D[2], 6);
	mid[12] = ROTL64(D[3], 25);
	mid[13] = ROTL64(D[4], 8);
	mid[14] = ROTL64(D[0], 18);
	mid[15] = ROTL64(message[4] ^ D[4], 27);
	mid[16] = ROTL64(message[5] ^ D[0], 36);
	mid[17] = ROTL64(D[1], 10);
	mid[18] = ROTL64(D[2], 15);
	mid[19] = ROTL64(D[3], 56);
	mid[20] = ROTL64(message[2] ^ D[2], 62);
	mid[21] = ROTL64(message[8] ^ D[3], 55);
	mid[22] = ROTL64(D[4], 39);
	mid[23] = ROTL64(D[0], 41);
	mid[24] = ROTL64(D[1], 2);

	state_t midState;
	std::memcpy(&midState, &mid, STATE_LENGTH);
	return midState;
}
