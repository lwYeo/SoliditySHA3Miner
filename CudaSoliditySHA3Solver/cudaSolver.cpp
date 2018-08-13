#include "cudasolver.h"
#pragma unmanaged

#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64 - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

bool CUDASolver::m_pause{ false };

bool CUDASolver::foundNvAPI64()
{
	return NvAPI::foundNvAPI64();
}

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

CUDASolver::CUDASolver(std::string const maxDifficulty, std::string kingAddress) noexcept :
	s_address{ "" },
	s_challenge{ "" },
	s_target{ "" },
	s_difficulty{ "" },
	m_address{ 0 },
	m_prefix{ 0 },
	m_miningMessage{ 0 },
	m_target{ 0 },
	m_difficulty{ 0 },
	m_maxDifficulty{ maxDifficulty },
	m_solutionTemplate{ new uint8_t[UINT256_LENGTH]{ 0 } },
	s_kingAddress{ kingAddress }
{
	if (NvAPI::foundNvAPI64()) NvAPI::initialize();

	m_solutionHashStartTime.store(std::chrono::steady_clock::now());
}

CUDASolver::~CUDASolver() noexcept
{
	stopFinding();

	free(m_solutionTemplate);

	NvAPI::unload();
}

void CUDASolver::setGetSolutionTemplateCallback(GetSolutionTemplateCallback solutionTemplateCallback)
{
	m_getSolutionTemplateCallback = solutionTemplateCallback;
}

void CUDASolver::setGetWorkPositionCallback(GetWorkPositionCallback workPositionCallback)
{
	m_getWorkPositionCallback = workPositionCallback;
}

void CUDASolver::setResetWorkPositionCallback(ResetWorkPositionCallback resetWorkPositionCallback)
{
	m_resetWorkPositionCallback = resetWorkPositionCallback;
}

void CUDASolver::setIncrementWorkPositionCallback(IncrementWorkPositionCallback incrementWorkPositionCallback)
{
	m_incrementWorkPositionCallback = incrementWorkPositionCallback;
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

	m_devices.push_back(std::make_unique<Device>(deviceID));
	auto& assignDevice = m_devices.back();

	assignDevice->name = deviceProp.name;
	assignDevice->computeVersion = deviceProp.major * 100 + deviceProp.minor * 10;

	std::string deviceName{ assignDevice->name };
	std::transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::toupper);

	if (assignDevice->computeVersion >= 500)
	{
		float defaultIntensity{ DEFALUT_INTENSITY };

		if (deviceName.find("1180") != std::string::npos || deviceName.find("1080") != std::string::npos
			|| deviceName.find("1070 TI") != std::string::npos || deviceName.find("1070TI") != std::string::npos)
			defaultIntensity = 29.01f;

		else if (deviceName.find("1070") != std::string::npos || deviceName.find("980") != std::string::npos)
			defaultIntensity = 28.0f;

		else if (deviceName.find("1060") != std::string::npos || deviceName.find("970") != std::string::npos)
			defaultIntensity = 27.0f;

		else if (deviceName.find("1050") != std::string::npos || deviceName.find("960") != std::string::npos)
			defaultIntensity = 26.0f;

		assignDevice->intensity = (intensity < 1.000f) ? defaultIntensity : intensity;
	}
	else assignDevice->intensity = (intensity < 1.000f) ? 24.0f : intensity; // For older GPUs

	onMessage(assignDevice->deviceID, "Info", "Assigned CUDA device (" + assignDevice->name + ")...");

	onMessage(assignDevice->deviceID, "Info", "Compute capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor));

	onMessage(assignDevice->deviceID, "Info", "Intensity: " + std::to_string(assignDevice->intensity));

	initializeDevice(assignDevice);

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

bool CUDASolver::isPaused()
{
	return m_pause;
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
	
	std::memcpy(&m_prefix, &tempPrefix, PREFIX_LENGTH);
	std::memcpy(&m_miningMessage, &m_prefix, PREFIX_LENGTH);

	getSolutionTemplate(m_solutionTemplate);
	std::memcpy(&m_miningMessage[PREFIX_LENGTH], m_solutionTemplate, UINT256_LENGTH);

	state_t tempMidState{ getMidState(m_miningMessage) };

	for (auto& device : m_devices)
	{
		if (device->deviceID < 0) continue;

		device->currentMidState = tempMidState;
		device->isNewMessage = true;
	}

	onMessage(-1, "Info", "New challenge detected " + s_challenge.substr(0, 18) + "...");
}

void CUDASolver::updateTarget(std::string const target)
{
	if (m_customDifficulty > 0u && !(target == (m_maxDifficulty / m_customDifficulty).GetHex())) return;

	arith_uint256 tempTarget = arith_uint256(target);
	if (tempTarget == m_target) return;

	s_target = (target.substr(0, 2) == "0x") ? target : "0x" + target;
	m_target = tempTarget;

	uint64_t tempHigh64Target{ std::stoull(s_target.substr(2).substr(0, UINT64_LENGTH * 2), nullptr, 16) };

	for (auto& device : m_devices)
	{
		if (device->deviceID < 0) continue;

		device->currentHigh64Target = tempHigh64Target;
		device->isNewTarget = true;
	}

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

void CUDASolver::setCustomDifficulty(uint32_t const customDifficulty)
{
	if (customDifficulty == 0u) return;

	s_customDifficulty = std::to_string(customDifficulty);
	m_customDifficulty = arith_uint256{ customDifficulty };

	onMessage(-1, "Info", "Custom difficulty (" + s_customDifficulty + ") detected...");
	updateTarget((m_maxDifficulty / m_customDifficulty).GetHex());
}

void CUDASolver::startFinding()
{
	uint64_t lastPosition;
	resetWorkPosition(lastPosition);
	m_solutionHashStartTime.store(std::chrono::steady_clock::now());

	for (auto& device : m_devices)
	{
		device->miningThread = std::thread(&CUDASolver::findSolution, this, device->deviceID);
		device->miningThread.detach();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void CUDASolver::stopFinding()
{
	for (auto& device : m_devices) device->mining = false;

	std::this_thread::sleep_for(std::chrono::seconds(1));
}

void CUDASolver::pauseFinding(bool pauseFinding)
{
	m_pause = pauseFinding;
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

int CUDASolver::getDeviceSettingMaxCoreClock(int deviceID)
{
	std::string errorMessage;
	int maxCoreClock;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getSettingMaxCoreClock(&maxCoreClock, &errorMessage))
				return maxCoreClock;

	return -1;
}

int CUDASolver::getDeviceSettingMaxMemoryClock(int deviceID)
{
	std::string errorMessage;
	int maxMemoryClock;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getSettingMaxMemoryClock(&maxMemoryClock, &errorMessage))
				return maxMemoryClock;

	return -1;
}

int CUDASolver::getDeviceSettingPowerLimit(int deviceID)
{
	std::string errorMessage;
	int powerLimit;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getSettingPowerLimit(&powerLimit, &errorMessage))
				return powerLimit;

	return -1;
}

int CUDASolver::getDeviceSettingThermalLimit(int deviceID)
{
	std::string errorMessage;
	int thermalLimit;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getSettingThermalLimit(&thermalLimit, &errorMessage))
				return thermalLimit;

	return -1;
}

int CUDASolver::getDeviceSettingFanLevelPercent(int deviceID)
{
	std::string errorMessage;
	int fanLevel;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getSettingFanLevelPercent(&fanLevel, &errorMessage))
				return fanLevel;
;

	return -1;
}

int CUDASolver::getDeviceCurrentFanTachometerRPM(int deviceID)
{
	std::string errorMessage;
	int currentReading;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentFanTachometerRPM(&currentReading, &errorMessage))
				return currentReading;

	return -1;
}

int CUDASolver::getDeviceCurrentTemperature(int deviceID)
{
	std::string errorMessage;
	int currentTemperature;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentTemperature(&currentTemperature, &errorMessage))
				return currentTemperature;

	return INT32_MIN;
}

int CUDASolver::getDeviceCurrentCoreClock(int deviceID)
{
	std::string errorMessage;
	int currentCoreClock;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentCoreClock(&currentCoreClock, &errorMessage))
				return currentCoreClock;

	return -1;
}

int CUDASolver::getDeviceCurrentMemoryClock(int deviceID)
{
	std::string errorMessage;
	int currentMemoryClock;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentMemoryClock(&currentMemoryClock, &errorMessage))
				return currentMemoryClock;

	return -1;
}

int CUDASolver::getDeviceCurrentUtilizationPercent(int deviceID)
{
	std::string errorMessage;
	int currentUtilization;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentUtilizationPercent(&currentUtilization, &errorMessage))
				return currentUtilization;

	return -1;
}

int CUDASolver::getDeviceCurrentPstate(int deviceID)
{
	std::string errorMessage;
	int currentPstate;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentPstate(&currentPstate, &errorMessage))
				return currentPstate;

	return -1;
}

std::string CUDASolver::getDeviceCurrentThrottleReasons(int deviceID)
{
	std::string errorMessage, throttleReasons;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentThrottleReasons(&throttleReasons, &errorMessage))
				return throttleReasons;

	return "";
}

// --------------------------------------------------------------------
// Private
// --------------------------------------------------------------------

void CUDASolver::getSolutionTemplate(uint8_t *&solutionTemplate)
{
	m_getSolutionTemplateCallback(solutionTemplate);
}

void CUDASolver::getWorkPosition(uint64_t &workPosition)
{
	m_getWorkPositionCallback(workPosition);
}

void CUDASolver::resetWorkPosition(uint64_t &lastPosition)
{
	m_resetWorkPositionCallback(lastPosition);
}

void CUDASolver::incrementWorkPosition(uint64_t &lastPosition, uint64_t increment)
{
	m_incrementWorkPositionCallback(lastPosition, increment);
}

void CUDASolver::onMessage(int deviceID, const char *type, const char *message)
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

void CUDASolver::onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device)
{
	if (!isSubmitStale && challenge != s_challenge)
		return;
	else if (isSubmitStale && challenge != s_challenge)
		onMessage(device->deviceID, "Warn", "GPU found stale solution, verifying...");
	else
		onMessage(device->deviceID, "Info", "GPU found solution, verifying...");

	prefix_t prefix;
	std::memcpy(&prefix, &m_miningMessage, PREFIX_LENGTH);

	byte32_t emptySolution;
	std::memset(&emptySolution, 0u, UINT256_LENGTH);
	if (solution == emptySolution)
	{
		onMessage(device->deviceID, "Error", "CPU verification failed: empty solution"
			+ std::string("\nChallenge: ") + challenge
			+ "\nAddress: " + s_address
			+ "\nTarget: " + s_target);
		return;
	}

	std::string prefixStr{ bytesToHexString(prefix) };
	std::string solutionStr{ bytesToHexString(solution) };

	std::string digestStr = keccak256(prefixStr + solutionStr);
	arith_uint256 digest = arith_uint256(digestStr);
	onMessage(device->deviceID, "Debug", "Digest: 0x" + digestStr);

	if (digest >= m_target)
	{
		std::string s_hi64Target{ s_target.substr(2).substr(0, UINT64_LENGTH * 2) };
		for (uint32_t i{ 0 }; i < (UINT256_LENGTH - UINT64_LENGTH); ++i) s_hi64Target += "00";
		arith_uint256 high64Target{ s_hi64Target };

		if (digest <= high64Target) // on rare ocassion where it falls in between m_target and high64Target
			onMessage(device->deviceID, "Warn", "CPU verification failed: invalid solution");
		else
		{
			onMessage(device->deviceID, "Error", "CPU verification failed: invalid solution"
				+ std::string("\nChallenge: ") + challenge
				+ "\nAddress: " + s_address
				+ "\nSolution: 0x" + solutionStr
				+ "\nDigest: 0x" + digestStr
				+ "\nTarget: " + s_target);
		}
	}
	else
	{
		onMessage(device->deviceID, "Info", "Solution verified by CPU, submitting nonce 0x" + solutionStr + "...");
		m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str(), m_customDifficulty > 0u);
	}
}

void CUDASolver::submitSolutions(std::set<uint64_t> solutions, std::string challenge, int const deviceID)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device) { return device->deviceID == deviceID; });

	getSolutionTemplate(m_solutionTemplate);

	for (uint64_t midStateSolution : solutions)
	{
		byte32_t solution{ 0 };
		std::memcpy(&solution, m_solutionTemplate, UINT256_LENGTH);

		if (s_kingAddress.empty())
			std::memcpy(&solution[12], &midStateSolution, UINT64_LENGTH); // keep first and last 12 bytes, fill middle 8 bytes for mid state
		else
			std::memcpy(&solution[ADDRESS_LENGTH], &midStateSolution, UINT64_LENGTH); // Shifted for King address

		onSolution(solution, challenge, device);
	}
}

uint64_t CUDASolver::getNextWorkPosition(std::unique_ptr<Device> &device)
{
	uint64_t lastPosition;
	incrementWorkPosition(lastPosition, device->threads());
	device->hashCount += device->threads();

	return lastPosition;
}

state_t const CUDASolver::getMidState(message_t &newMessage)
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
