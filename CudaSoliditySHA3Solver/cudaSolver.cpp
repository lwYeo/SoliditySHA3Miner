#include "cudaErrorCheck.cu"
#include "cudasolver.h"
#pragma unmanaged

#define ROTL64(x, y) (((x) << (y)) ^ ((x) >> (64u - (y))))
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64u - (y))))

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

bool CUDASolver::m_pause{ false };
bool CUDASolver::m_isSubmitting{ false };
bool CUDASolver::m_isKingMaking{ false };

bool CUDASolver::foundNvAPI64()
{
	return NV_API::foundNvAPI64();
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

CUDASolver::CUDASolver() noexcept :
	s_address{ "" },
	s_challenge{ "" },
	s_target{ "" },
	m_address{ 0 },
	m_kingAddress{ 0 },
	m_miningMessage{ 0 },
	m_target{ 0 }
{
	try { if (NV_API::foundNvAPI64()) NV_API::initialize(); }
	catch (std::exception ex) { onMessage(-1, "Error", ex.what()); }
}

CUDASolver::~CUDASolver() noexcept
{
	stopFinding();

	try { NV_API::unload(); }
	catch (...) {}
}

void CUDASolver::setGetKingAddressCallback(GetKingAddressCallback kingAddressCallback)
{
	m_getKingAddressCallback = kingAddressCallback;
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

bool CUDASolver::assignDevice(int const deviceID, float &intensity)
{
	onMessage(deviceID, "Info", "Assigning device...");

	getKingAddress(&m_kingAddress);
	m_isKingMaking = (!isAddressEmpty(m_kingAddress));

	struct cudaDeviceProp deviceProp;
	cudaError_t response = cudaGetDeviceProperties(&deviceProp, deviceID);
	if (response != cudaSuccess)
	{
		onMessage(deviceID, "Error", cudaGetErrorString(response));
		return false;
	}

	m_devices.push_back(std::make_unique<Device>(deviceID));
	auto &assignDevice = m_devices.back();

	assignDevice->name = deviceProp.name;
	assignDevice->computeVersion = deviceProp.major * 100 + deviceProp.minor * 10;

	std::string deviceName{ assignDevice->name };
	std::transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::toupper);

	if (assignDevice->computeVersion >= 500)
	{
		float defaultIntensity{ DEFALUT_INTENSITY };

		if (m_isKingMaking)
		{
			if (deviceName.find("2080") != std::string::npos || deviceName.find("2070") != std::string::npos
				|| deviceName.find("1080 TI") != std::string::npos || deviceName.find("1080TI") != std::string::npos
				|| deviceName.find("1080") != std::string::npos)
				defaultIntensity = 27.54f;

			else if (deviceName.find("2060") != std::string::npos
				|| deviceName.find("1070 TI") != std::string::npos || deviceName.find("1070TI") != std::string::npos)
				defaultIntensity = 27.46f;

			else if (deviceName.find("2050") != std::string::npos
				|| deviceName.find("1070") != std::string::npos || deviceName.find("980") != std::string::npos)
				defaultIntensity = 27.01f;

			else if (deviceName.find("1060") != std::string::npos || deviceName.find("970") != std::string::npos)
				defaultIntensity = 26.01f;

			else if (deviceName.find("1050") != std::string::npos || deviceName.find("960") != std::string::npos)
				defaultIntensity = 25.01f;
		}
		else
		{
			if (deviceName.find("2080") != std::string::npos || deviceName.find("2070") != std::string::npos || deviceName.find("1080") != std::string::npos
				|| deviceName.find("1070 TI") != std::string::npos || deviceName.find("1070TI") != std::string::npos)
				defaultIntensity = 29.01f;

			else if (deviceName.find("2060") != std::string::npos || deviceName.find("1070") != std::string::npos || deviceName.find("980") != std::string::npos)
				defaultIntensity = 28.0f;

			else if (deviceName.find("2050") != std::string::npos || deviceName.find("1060") != std::string::npos || deviceName.find("970") != std::string::npos)
				defaultIntensity = 27.0f;

			else if (deviceName.find("1050") != std::string::npos || deviceName.find("960") != std::string::npos)
				defaultIntensity = 26.0f;
		}

		assignDevice->intensity = (intensity < 1.000f) ? defaultIntensity : intensity;
	}
	else assignDevice->intensity = (intensity < 1.000f) ? 24.0f : intensity; // For older GPUs

	intensity = assignDevice->intensity;

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
	assert(prefix.length() == ((UINT256_LENGTH + ADDRESS_LENGTH) * 2 + 2));

	s_challenge = prefix.substr(0, 2 + UINT256_LENGTH * 2);
	s_address = "0x" + prefix.substr(2 + UINT256_LENGTH * 2, ADDRESS_LENGTH * 2);

	getSolutionTemplate(&m_solutionTemplate);

	hexStringToBytes(s_challenge, m_miningMessage.structure.challenge);
	hexStringToBytes(s_address, m_miningMessage.structure.address);
	m_miningMessage.structure.solution = m_solutionTemplate;

	sponge_ut midState{ getMidState(m_miningMessage) };

	for (auto& device : m_devices)
	{
		if (device->deviceID < 0) continue;

		device->currentMessage = m_miningMessage;
		device->currentMidstate = midState;
		device->isNewMessage = true;
	}
}

void CUDASolver::updateTarget(std::string const target)
{
	arith_uint256 tempTarget = arith_uint256(target);
	if (tempTarget == m_target) return;

	s_target = (target.substr(0, 2) == "0x") ? target : "0x" + target;
	m_target = tempTarget;

	byte32_t bTarget;
	hexStringToBytes(s_target, bTarget);

	uint64_t tempHigh64Target{ std::stoull(s_target.substr(2).substr(0, UINT64_LENGTH * 2), nullptr, 16) };

	for (auto& device : m_devices)
	{
		if (device->deviceID < 0) continue;

		device->currentTarget = bTarget;
		device->currentHigh64Target = tempHigh64Target;
		device->isNewTarget = true;
	}
}

void CUDASolver::startFinding()
{
	onMessage(-1, "Info", "Start mining...");

	for (auto& device : m_devices)
	{
		if (m_isKingMaking)
			device->miningThread = std::thread(&CUDASolver::findSolutionKing, this, device->deviceID);
		else
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
	int tachometerRPM;

	for (auto& device : m_devices)
		if (device->deviceID == deviceID)
			if (device->getCurrentFanTachometerRPM(&tachometerRPM, &errorMessage))
				return tachometerRPM;

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

bool CUDASolver::isAddressEmpty(address_t &address)
{
	for (uint32_t i{ 0 }; i < ADDRESS_LENGTH; ++i)
		if (address[i] > 0u) return false;

	return true;
}

void CUDASolver::getKingAddress(address_t *kingAddress)
{
	m_getKingAddressCallback(kingAddress->data());
}

void CUDASolver::getSolutionTemplate(byte32_t *solutionTemplate)
{
	m_getSolutionTemplateCallback(solutionTemplate->data());
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

void CUDASolver::onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device)
{
	if (!isSubmitStale && challenge != s_challenge)
		return;
	else if (isSubmitStale && challenge != s_challenge)
		onMessage(device->deviceID, "Warn", "GPU found stale solution, verifying...");
	else
		onMessage(device->deviceID, "Info", "GPU found solution, verifying...");

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

	message_ut newMessage = m_miningMessage;
	newMessage.structure.solution = solution;

	std::string newMessageStr{ bytesToHexString(newMessage.byteArray) };
	std::string solutionStr{ bytesToHexString(solution) };

	byte32_t bDigest;
	keccak_256(&bDigest[0], UINT256_LENGTH, &newMessage.byteArray[0], MESSAGE_LENGTH);

	std::string digestStr = bytesToHexString(bDigest);
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
		onMessage(device->deviceID, "Debug", std::string{ "Solution details..." }
			+ "\nChallenge: " + challenge
			+ "\nAddress: " + s_address
			+ "\nSolution: 0x" + solutionStr
			+ "\nDigest: 0x" + digestStr
			+ "\nTarget: " + s_target);
		m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), challenge.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str());
	}
}

void CUDASolver::submitSolutions(std::set<uint64_t> solutions, std::string challenge, int const deviceID)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device) { return device->deviceID == deviceID; });

	while (m_isSubmitting)
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	m_isSubmitting = true;

	getSolutionTemplate(&m_solutionTemplate);

	for (uint64_t midStateSolution : solutions)
	{
		byte32_t solution{ 0 };
		std::memcpy(&solution, &m_solutionTemplate, UINT256_LENGTH);

		if (m_isKingMaking)
			std::memcpy(&solution[ADDRESS_LENGTH], &midStateSolution, UINT64_LENGTH); // Shifted for King address
		else
			std::memcpy(&solution[12], &midStateSolution, UINT64_LENGTH); // keep first and last 12 bytes, fill middle 8 bytes for mid state

		onSolution(solution, challenge, device);
	}
	m_isSubmitting = false;
}

uint64_t CUDASolver::getNextWorkPosition(std::unique_ptr<Device> &device)
{
	uint64_t lastPosition;
	incrementWorkPosition(lastPosition, device->threads());
	device->hashCount += device->threads();

	return lastPosition;
}

sponge_ut const CUDASolver::getMidState(message_ut &newMessage)
{
	sponge_ut midstate;
	uint64_t C[5], D[5];
	uint64_t message[11]{ 0 };

	std::memcpy(&message, &newMessage, MESSAGE_LENGTH);

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

	midstate.uint64Array[0] = message[0] ^ D[0];
	midstate.uint64Array[1] = ROTL64(message[6] ^ D[1], 44);
	midstate.uint64Array[2] = ROTL64(D[2], 43);
	midstate.uint64Array[3] = ROTL64(D[3], 21);
	midstate.uint64Array[4] = ROTL64(D[4], 14);
	midstate.uint64Array[5] = ROTL64(message[3] ^ D[3], 28);
	midstate.uint64Array[6] = ROTL64(message[9] ^ D[4], 20);
	midstate.uint64Array[7] = ROTL64(message[10] ^ D[0] ^ 0x100000000ull, 3);
	midstate.uint64Array[8] = ROTL64(0x8000000000000000ull ^ D[1], 45);
	midstate.uint64Array[9] = ROTL64(D[2], 61);
	midstate.uint64Array[10] = ROTL64(message[1] ^ D[1], 1);
	midstate.uint64Array[11] = ROTL64(message[7] ^ D[2], 6);
	midstate.uint64Array[12] = ROTL64(D[3], 25);
	midstate.uint64Array[13] = ROTL64(D[4], 8);
	midstate.uint64Array[14] = ROTL64(D[0], 18);
	midstate.uint64Array[15] = ROTL64(message[4] ^ D[4], 27);
	midstate.uint64Array[16] = ROTL64(message[5] ^ D[0], 36);
	midstate.uint64Array[17] = ROTL64(D[1], 10);
	midstate.uint64Array[18] = ROTL64(D[2], 15);
	midstate.uint64Array[19] = ROTL64(D[3], 56);
	midstate.uint64Array[20] = ROTL64(message[2] ^ D[2], 62);
	midstate.uint64Array[21] = ROTL64(message[8] ^ D[3], 55);
	midstate.uint64Array[22] = ROTL64(D[4], 39);
	midstate.uint64Array[23] = ROTL64(D[0], 41);
	midstate.uint64Array[24] = ROTL64(D[1], 2);

	return midstate;
}

void CUDASolver::initializeDevice(std::unique_ptr<Device> &device)
{
	auto deviceID = device->deviceID;

	if (!device->initialized)
	{
		onMessage(deviceID, "Info", "Initializing device...");
		CudaSafeCall(cudaSetDevice(deviceID));

		device->h_Solutions = reinterpret_cast<uint64_t *>(malloc(MAX_SOLUTION_COUNT_DEVICE * UINT64_LENGTH));
		device->h_SolutionCount = reinterpret_cast<uint32_t *>(malloc(UINT32_LENGTH));
		std::memset(device->h_Solutions, 0u, MAX_SOLUTION_COUNT_DEVICE * UINT64_LENGTH);
		std::memset(device->h_SolutionCount, 0u, UINT32_LENGTH);

		CudaSafeCall(cudaDeviceReset());
		CudaSafeCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost));

		CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&device->h_SolutionCount), UINT32_LENGTH, cudaHostAllocMapped));
		CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&device->h_Solutions), MAX_SOLUTION_COUNT_DEVICE * UINT64_LENGTH, cudaHostAllocMapped));

		CudaSafeCall(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device->d_SolutionCount), reinterpret_cast<void *>(device->h_SolutionCount), 0));
		CudaSafeCall(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device->d_Solutions), reinterpret_cast<void *>(device->h_Solutions), 0));

		device->initialized = true;

		if (NV_API::foundNvAPI64())
		{
			std::string errorMessage;
			int maxCoreClock, maxMemoryClock, powerLimit, thermalLimit, fanLevel;

			if (device->getSettingMaxCoreClock(&maxCoreClock, &errorMessage))
				onMessage(device->deviceID, "Info", "Max core clock setting: " + std::to_string(maxCoreClock) + "MHz.");
			else
				onMessage(device->deviceID, "Error", "Failed to get max core clock setting: " + errorMessage);

			if (device->getSettingMaxMemoryClock(&maxMemoryClock, &errorMessage))
				onMessage(device->deviceID, "Info", "Max memory clock setting: " + std::to_string(maxMemoryClock) + "MHz.");
			else
				onMessage(device->deviceID, "Error", "Failed to get max memory clock setting: " + errorMessage);

			if (device->getSettingPowerLimit(&powerLimit, &errorMessage))
				onMessage(device->deviceID, "Info", "Power limit setting: " + std::to_string(powerLimit) + "%.");
			else
				onMessage(device->deviceID, "Error", "Failed to get power limit setting: " + errorMessage);

			if (device->getSettingThermalLimit(&thermalLimit, &errorMessage))
				onMessage(device->deviceID, "Info", "Thermal limit setting: " + std::to_string(thermalLimit) + "C.");
			else
				onMessage(device->deviceID, "Error", "Failed to get thermal limit setting: " + errorMessage);

			if (device->getSettingFanLevelPercent(&fanLevel, &errorMessage))
				onMessage(device->deviceID, "Info", "Fan level setting: " + std::to_string(fanLevel) + "%.");
			else
				onMessage(device->deviceID, "Error", "Failed to get fan level setting: " + errorMessage);
		}
	}
}

void CUDASolver::checkInputs(std::unique_ptr<Device>& device, char *currentChallenge)
{
	if (device->isNewMessage || device->isNewTarget)
	{
		uint64_t lastPosition;
		getWorkPosition(lastPosition);

		for (auto& device : m_devices)
		{
			if (device->hashCount.load() > INT64_MAX)
			{
				device->hashCount.store(0ull);
				device->hashStartTime.store(std::chrono::steady_clock::now());
			}
		}

		if (device->isNewTarget)
		{
			if (m_isKingMaking)
				pushTargetKing(device);
			else
				pushTarget(device);
		}

		if (device->isNewMessage)
		{
			if (m_isKingMaking)
				pushMessageKing(device);
			else
				pushMessage(device);

			strcpy_s(currentChallenge, s_challenge.size() + 1, s_challenge.c_str());
		}
	}
}
