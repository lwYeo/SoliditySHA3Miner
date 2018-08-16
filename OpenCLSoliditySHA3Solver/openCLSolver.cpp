#include "openCLSolver.h"
#pragma unmanaged

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

std::vector<openCLSolver::Platform> openCLSolver::platforms;
bool openCLSolver::m_pause{ false };

void openCLSolver::preInitialize(bool allowIntel, std::string &errorMessage)
{
	cl_int status{ CL_SUCCESS };
	cl_uint numPlatforms{ 0 };
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	
	if (status != CL_SUCCESS)
	{
		errorMessage = "No OpenCL platforms available.";
		return;
	}

	cl_platform_id* tempPlatforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, tempPlatforms, NULL);

	if (status != CL_SUCCESS)
	{
		errorMessage = "Failed to get OpenCL platforms.";
		return;
	}

	for (unsigned int i{ 0 }; i < numPlatforms; ++i)
	{
		char platformNameBuf[256];
		status = clGetPlatformInfo(tempPlatforms[i], CL_PLATFORM_NAME, sizeof(platformNameBuf), platformNameBuf, NULL);
		std::string platformName{ (std::string{ platformNameBuf } == "") ? "Unknown" : platformNameBuf };

		std::string tempPlatform{ platformName };
		std::transform(tempPlatform.begin(), tempPlatform.end(), tempPlatform.begin(), ::toupper);

		if (tempPlatform.find("ACCELERATED PARALLEL PROCESSING") != std::string::npos 
			|| (allowIntel && tempPlatform.find("INTEL") != std::string::npos))
		{
			platforms.emplace_back(Platform{ tempPlatforms[i], platformName });
		}
	}

	Device::preInitialize(errorMessage);
}

bool openCLSolver::foundAdlApi()
{
	return ADL_API::foundAdlApi();
}

std::string openCLSolver::getPlatformNames()
{
	std::string platformNames{ "" };
	for (auto& platform : platforms)
	{
		if (!platformNames.empty()) platformNames += "\n";
		platformNames += platform.name;
	}
	return platformNames;
}

int openCLSolver::getDeviceCount(std::string platformName, std::string &errorMessage)
{
	errorMessage = "";
	cl_int status{ CL_SUCCESS };
	std::vector<cl::Device> devices;

	for (auto& platform : platforms)
	{
		if (platform.name == platformName)
		{
			cl_uint deviceCount;
			status = clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, NULL, NULL, &deviceCount);

			if (status != CL_SUCCESS)
			{
				errorMessage = "Unable to get device count for " + platformName;
				continue;
			}

			return deviceCount;
		}
	}
	return 0;
}

std::string openCLSolver::getDeviceName(std::string platformName, int deviceEnum, std::string &errorMessage)
{
	errorMessage = "";
	cl_int status{ CL_SUCCESS };
	std::string deviceName = "";

	for (auto& platform : platforms)
	{
		if (platform.name == platformName)
		{
			cl_uint deviceCount;
			status = clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, NULL, NULL, &deviceCount);

			if (status != CL_SUCCESS)
			{
				errorMessage = "Failed to get device count for " + platformName;
				continue;
			}

			cl_device_id* deviceIDs = new cl_device_id[deviceCount];
			status = clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, deviceCount, deviceIDs, NULL);

			if (status != CL_SUCCESS)
			{
				errorMessage = "Failed to get device IDs for " + platformName;
				continue;
			}

			char tempDeviceName[256];
			status = clGetDeviceInfo(deviceIDs[deviceEnum], CL_DEVICE_NAME, sizeof(tempDeviceName), tempDeviceName, NULL);

			std::string deviceName{ tempDeviceName };

			return deviceName.empty() ? "Unknown" : deviceName;
		}
	}
	return "Unknown";
}

// --------------------------------------------------------------------
// Public
// --------------------------------------------------------------------

openCLSolver::openCLSolver(std::string const maxDifficulty, std::string kingAddress) noexcept :
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
	try { if (ADL_API::foundAdlApi()) ADL_API::initialize(); }
	catch (std::exception ex) { onMessage("", -1, "Error", ex.what()); }

	m_solutionHashStartTime.store(std::chrono::steady_clock::now());
}

openCLSolver::~openCLSolver() noexcept
{
	stopFinding();

	free(m_solutionTemplate);
}

void openCLSolver::setGetSolutionTemplateCallback(GetSolutionTemplateCallback solutionTemplateCallback)
{
	m_getSolutionTemplateCallback = solutionTemplateCallback;
}

void openCLSolver::setGetWorkPositionCallback(GetWorkPositionCallback workPositionCallback)
{
	m_getWorkPositionCallback = workPositionCallback;
}

void openCLSolver::setResetWorkPositionCallback(ResetWorkPositionCallback resetWorkPositionCallback)
{
	m_resetWorkPositionCallback = resetWorkPositionCallback;
}

void openCLSolver::setIncrementWorkPositionCallback(IncrementWorkPositionCallback incrementWorkPositionCallback)
{
	m_incrementWorkPositionCallback = incrementWorkPositionCallback;
}

void openCLSolver::setMessageCallback(MessageCallback messageCallback)
{
	m_messageCallback = messageCallback;
}

void openCLSolver::setSolutionCallback(SolutionCallback solutionCallback)
{
	m_solutionCallback = solutionCallback;
}

bool openCLSolver::isAssigned()
{
	for (auto& device : m_devices)
		if (device->deviceEnum > -1)
			return true;

	return false;
}

bool openCLSolver::isAnyInitialised()
{
	for (auto& device : m_devices)
		if (device->deviceEnum > -1 && device->initialized)
			return true;

	return false;
}

bool openCLSolver::isMining()
{
	for (auto& device : m_devices)
		if (device->mining)
			return true;

	return false;
}

bool openCLSolver::isPaused()
{
	return m_pause;
}

bool openCLSolver::assignDevice(std::string platformName, int deviceEnum, float const intensity)
{
	cl_int status{ CL_SUCCESS };

	for (auto& platform : platforms)
	{
		if (platform.name == platformName)
		{
			cl_uint deviceCount;
			status = clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, NULL, NULL, &deviceCount);

			if (status != CL_SUCCESS)
			{
				onMessage(platformName.c_str() , -1, "Error", "Failed to get device count.");
				continue;
			}

			cl_device_id* deviceIDs = new cl_device_id[deviceCount];
			status = clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, deviceCount, deviceIDs, NULL);

			if (status != CL_SUCCESS)
			{
				onMessage(platformName.c_str(), -1, "Error", "Failed to get device IDs");
				continue;
			}

			onMessage(platformName.c_str(), deviceEnum, "Info", "Assigning OpenCL device...");
			try
			{
				m_devices.emplace_back(new Device(deviceEnum, deviceIDs[deviceEnum], CL_DEVICE_TYPE_GPU, platform.id, intensity, 0));

				auto &assignDevice = m_devices.back();

				onMessage(platformName.c_str(), deviceEnum, "Info", "Assigned OpenCL device (" + assignDevice->name + ")...");
				onMessage(platformName.c_str(), deviceEnum, "Info", "Intensity: " + std::to_string(assignDevice->userDefinedIntensity));

				if (assignDevice->isAPP() && foundAdlApi())
				{
					std::string errorMessage;
					int maxCoreClock, maxMemoryClock, powerLimit, thermalLimit, fanLevel;

					if (assignDevice->getSettingMaxCoreClock(&maxCoreClock, &errorMessage))
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Info", "Max core clock setting: " + std::to_string(maxCoreClock) + "MHz.");
					else
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Error", "Failed to get max core clock setting: " + errorMessage);

					if (assignDevice->getSettingMaxMemoryClock(&maxMemoryClock, &errorMessage))
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Info", "Max memory clock setting: " + std::to_string(maxMemoryClock) + "MHz.");
					else
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Error", "Failed to get max memory clock setting: " + errorMessage);

					if (assignDevice->getSettingPowerLimit(&powerLimit, &errorMessage))
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Info", "Power limit setting: " + std::to_string(powerLimit) + "%.");
					else
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Error", "Failed to get power limit setting: " + errorMessage);

					if (assignDevice->getSettingThermalLimit(&thermalLimit, &errorMessage))
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Info", "Thermal limit setting: " + std::to_string(thermalLimit) + "C.");
					else
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Error", "Failed to get thermal limit setting: " + errorMessage);

					if (assignDevice->getSettingFanLevelPercent(&fanLevel, &errorMessage))
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Info", "Fan level setting: " + std::to_string(fanLevel) + "%.");
					else
						onMessage(platformName.c_str(), assignDevice->deviceEnum, "Error", "Failed to get fan level setting: " + errorMessage);
				}

				return true;
			}
			catch (std::exception ex)
			{
				onMessage(platformName.c_str(), deviceEnum, "Error", ex.what());
			}
		}
	}
	onMessage(platformName.c_str(), -1, "Error", "Failed to get device " + std::to_string(deviceEnum));

	return false;
}

void openCLSolver::updatePrefix(std::string const prefix)
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

	onMessage("", -1, "Info", "New challenge detected " + s_challenge.substr(0, 18) + "...");
}

void openCLSolver::updateTarget(std::string const target)
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

		device->currentHigh64Target[0] = tempHigh64Target;
		device->isNewTarget = true;
	}

	onMessage("", -1, "Info", "New target detected " + s_target.substr(0, 18) + "...");
}

void openCLSolver::updateDifficulty(std::string const difficulty)
{
	if (m_customDifficulty > 0u) return;

	arith_uint256 oldDifficulity{ m_difficulty };
	s_difficulty = difficulty;
	m_difficulty = arith_uint256(difficulty);

	if ((m_maxDifficulty != 0ull) && (m_difficulty != oldDifficulity))
	{
		onMessage("", -1, "Info", "New difficulity detected (" + std::to_string(m_difficulty.GetLow64()) + ")...");
		updateTarget((m_maxDifficulty / m_difficulty).GetHex());
	}
}

void openCLSolver::setCustomDifficulty(uint32_t customDifficulty)
{
	if (customDifficulty == 0u) return;

	s_customDifficulty = std::to_string(customDifficulty);
	m_customDifficulty = arith_uint256{ customDifficulty };

	onMessage("", -1, "Info", "Custom difficulty (" + s_customDifficulty + ") detected...");
	updateTarget((m_maxDifficulty / m_customDifficulty).GetHex());
}

uint64_t openCLSolver::getTotalHashRate()
{
	uint64_t totalHashRate{ 0ull };
	for (auto& device : m_devices)
		totalHashRate += device->hashRate();

	return totalHashRate;
}

uint64_t openCLSolver::getHashRateByDevice(std::string platformName, int const deviceEnum)
{
	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			return device->hashRate();

	return 0ull;
}

int openCLSolver::getDeviceSettingMaxCoreClock(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int settingMaxCoreClock{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getSettingMaxCoreClock(&settingMaxCoreClock, &errorMessage);

	return settingMaxCoreClock;
}

int openCLSolver::getDeviceSettingMaxMemoryClock(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int settingMaxMemoryClock{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getSettingMaxMemoryClock(&settingMaxMemoryClock, &errorMessage);

	return settingMaxMemoryClock;
}

int openCLSolver::getDeviceSettingPowerLimit(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int settingPowerLimit{ INT32_MIN };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getSettingPowerLimit(&settingPowerLimit, &errorMessage);

	return settingPowerLimit;
}

int openCLSolver::getDeviceSettingThermalLimit(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int settingThermalLimit{ INT32_MIN };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getSettingThermalLimit(&settingThermalLimit, &errorMessage);

	return settingThermalLimit;
}

int openCLSolver::getDeviceSettingFanLevelPercent(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int fanLevelPercent{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getSettingFanLevelPercent(&fanLevelPercent, &errorMessage);

	return fanLevelPercent;
}

int openCLSolver::getDeviceCurrentFanTachometerRPM(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int fanTachometerRPM{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getCurrentFanTachometerRPM(&fanTachometerRPM, &errorMessage);

	return fanTachometerRPM;
}

int openCLSolver::getDeviceCurrentTemperature(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int temperature{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getCurrentTemperature(&temperature, &errorMessage);

	return temperature;
}

int openCLSolver::getDeviceCurrentCoreClock(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int coreClock{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getCurrentCoreClock(&coreClock, &errorMessage);

	return coreClock;
}

int openCLSolver::getDeviceCurrentMemoryClock(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int memoryClock{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getCurrentMemoryClock(&memoryClock, &errorMessage);

	return memoryClock;
}

int openCLSolver::getDeviceCurrentUtilizationPercent(std::string platformName, int deviceEnum)
{
	std::string errorMessage;
	int utilizationPercent{ -1 };

	for (auto& device : m_devices)
		if (device->platformName == platformName && device->deviceEnum == deviceEnum)
			device->getCurrentUtilizationPercent(&utilizationPercent, &errorMessage);

	return utilizationPercent;
}


void openCLSolver::startFinding()
{
	uint64_t lastPosition;
	resetWorkPosition(lastPosition);
	m_solutionHashStartTime.store(std::chrono::steady_clock::now());

	for (auto& device : m_devices)
	{
		onMessage(device->platformName, device->deviceEnum, "Info", "Initializing device...");

		std::string errorMessage;
		device->initialize(errorMessage);
		if (!device->initialized)
		{
			if (errorMessage != "") onMessage(device->platformName, device->deviceEnum, "Error", errorMessage);
			else onMessage(device->platformName, device->deviceEnum, "Error", "Failed to initialize device.");
		}
	}

	for (auto& device : m_devices)
	{
		device->miningThread = std::thread(&openCLSolver::findSolution, this, device->platformName, device->deviceEnum);
		device->miningThread.detach();
	}
}

void openCLSolver::stopFinding()
{
	for (auto& device : m_devices) device->mining = false;
	std::this_thread::sleep_for(std::chrono::seconds(1));
}

void openCLSolver::pauseFinding(bool pauseFinding)
{
	m_pause = pauseFinding;
}

// --------------------------------------------------------------------
// Private
// --------------------------------------------------------------------

void openCLSolver::getSolutionTemplate(uint8_t *&solutionTemplate)
{
	m_getSolutionTemplateCallback(solutionTemplate);
}

void openCLSolver::getWorkPosition(uint64_t &workPosition)
{
	m_getWorkPositionCallback(workPosition);
}

void openCLSolver::resetWorkPosition(uint64_t &lastPosition)
{
	m_resetWorkPositionCallback(lastPosition);
}

void openCLSolver::incrementWorkPosition(uint64_t &lastPosition, uint64_t increment)
{
	m_incrementWorkPositionCallback(lastPosition, increment);
}

void openCLSolver::onMessage(std::string platformName, int deviceEnum, std::string type, std::string message)
{
	m_messageCallback(platformName.empty() ? "OpenCL" : (platformName + " (OpenCL)").c_str(), deviceEnum, type.c_str(), message.c_str());
}

const std::string openCLSolver::keccak256(std::string const message)
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

void openCLSolver::onSolution(byte32_t const solution, std::string challenge, std::unique_ptr<Device> &device)
{
	if (!isSubmitStale && challenge != s_challenge)
		return;
	else if (isSubmitStale && challenge != s_challenge)
		onMessage(device->platformName, device->deviceEnum, "Warn", "GPU found stale solution, verifying...");
	else
		onMessage(device->platformName, device->deviceEnum, "Info", "GPU found solution, verifying...");

	prefix_t prefix;
	std::memcpy(&prefix, &m_miningMessage, PREFIX_LENGTH);

	byte32_t emptySolution;
	std::memset(&emptySolution, 0u, UINT256_LENGTH);
	if (solution == emptySolution)
	{
		onMessage(device->platformName, device->deviceEnum, "Error", "CPU verification failed: empty solution"
			+ std::string("\nChallenge: ") + challenge
			+ "\nAddress: " + s_address
			+ "\nTarget: " + s_target);
		return;
	}

	std::string prefixStr{ bytesToHexString(prefix) };
	std::string solutionStr{ bytesToHexString(solution) };

	std::string digestStr = keccak256(prefixStr + solutionStr);
	arith_uint256 digest = arith_uint256(digestStr);
	onMessage(device->platformName, device->deviceEnum, "Debug", "Digest: 0x" + digestStr);

	if (digest >= m_target)
	{
		std::string s_hi64Target{ s_target.substr(2).substr(0, UINT64_LENGTH * 2) };
		for (uint32_t i{ 0 }; i < (UINT256_LENGTH - UINT64_LENGTH); ++i) s_hi64Target += "00";
		arith_uint256 high64Target{ s_hi64Target };

		if (digest <= high64Target) // on rare ocassion where it falls in between m_target and high64Target
			onMessage(device->platformName, device->deviceEnum, "Warn", "CPU verification failed: invalid solution");
		else
		{
			onMessage(device->platformName, device->deviceEnum,  "Error", "CPU verification failed: invalid solution"
				+ std::string("\nChallenge: ") + challenge
				+ "\nAddress: " + s_address
				+ "\nSolution: 0x" + solutionStr
				+ "\nDigest: 0x" + digestStr
				+ "\nTarget: " + s_target);
		}
	}
	else
	{
		onMessage(device->platformName, device->deviceEnum, "Info", "Solution verified by CPU, submitting nonce 0x" + solutionStr + "...");
		m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str(), m_customDifficulty > 0u);
	}
}

void openCLSolver::submitSolutions(std::set<uint64_t> solutions, std::string challenge, std::string platformName, int const deviceEnum)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device)
	{
		return device->platformName == platformName && device->deviceEnum == deviceEnum;
	});

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

state_t const openCLSolver::getMidState(message_t &newMessage)
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

uint64_t const openCLSolver::getNextWorkPosition(std::unique_ptr<Device> &device)
{
	uint64_t lastPosition;
	incrementWorkPosition(lastPosition, device->globalWorkSize);
	device->hashCount += device->globalWorkSize;

	return lastPosition;
}

void openCLSolver::pushTarget(std::unique_ptr<Device> &device)
{
	device->status = clEnqueueWriteBuffer(device->queue, device->targetBuffer, CL_TRUE, 0u, UINT64_LENGTH, device->currentHigh64Target, NULL, NULL, NULL);
	if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting target buffer to kernel (" } +Device::getOpenCLErrorCodeStr(device->status) + ")...");

	device->isNewTarget = false;
}

void openCLSolver::pushMessage(std::unique_ptr<Device> &device)
{
	device->status = clEnqueueWriteBuffer(device->queue, device->midstateBuffer, CL_TRUE, 0u, STATE_LENGTH, device->currentMidState.data(), NULL, NULL, NULL);
	if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error writing to midstate buffer (" } +Device::getOpenCLErrorCodeStr(device->status) + ")...");

	device->isNewMessage = false;
}

void openCLSolver::checkInputs(std::unique_ptr<Device> &device, char *currentChallenge)
{
	if (device->isNewMessage || device->isNewTarget)
	{
		for (auto& device : m_devices)
		{
			if (device->hashCount.load() > INT64_MAX)
			{
				device->hashCount.store(0ull);
				device->hashStartTime.store(std::chrono::steady_clock::now());
			}
		}
		uint64_t lastPosition;
		resetWorkPosition(lastPosition);
		m_solutionHashStartTime.store(std::chrono::steady_clock::now());

		if (device->isNewTarget) pushTarget(device);

		if (device->isNewMessage)
		{
			pushMessage(device);
			strcpy_s(currentChallenge, s_challenge.size() + 1, s_challenge.c_str());
		}
	}
}

void openCLSolver::findSolution(std::string platformName, int const deviceEnum)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device)
	{
		return device->platformName == platformName && device->deviceEnum == deviceEnum;
	});
	
	if (!device->initialized) return;

	bool isCUDAorIntel = device->isCUDA() | device->isINTEL(); // cache value here

	onMessage(device->platformName, device->deviceEnum, "Info", "Start mining...");
	onMessage(device->platformName, device->deviceEnum, "Debug", "Threads: " + std::to_string(device->globalWorkSize) + " Local work size: " + std::to_string(device->localWorkSize) + " Block size:" + std::to_string(device->globalWorkSize / device->localWorkSize));

	device->mining = true;
	device->hashCount.store(0ull);
	device->hashStartTime.store(std::chrono::steady_clock::now() - std::chrono::milliseconds(200)); // reduce excessive high hashrate reporting at start

	uint64_t workPosition[MAX_WORK_POSITION_STORE];
	char *c_currentChallenge = (char *)malloc(s_challenge.size());
	do
	{
		while (m_pause) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }

		checkInputs(device, c_currentChallenge);

		for (uint32_t q{ 0 }; q < MAX_WORK_POSITION_STORE; ++q)
		{
			workPosition[q] = getNextWorkPosition(device);

			device->status = clSetKernelArg(device->kernel, 2u, UINT64_LENGTH, &workPosition[q]);
			if (device->status != CL_SUCCESS)
				onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting work positon buffer to kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

			device->status = clEnqueueNDRangeKernel(device->queue, device->kernel, 1u, NULL, &device->globalWorkSize, &device->localWorkSize, NULL, NULL, &device->kernelWaitEvent);
			if (device->status != CL_SUCCESS)
				onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error starting kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
		}

		if (isCUDAorIntel) // CUDA and Intel 100% CPU workaround
		{
			uint32_t waitStatus{ CL_QUEUED }, waitKernelCount{ 0u };
			while (waitStatus != CL_COMPLETE)
			{
				std::this_thread::sleep_for(std::chrono::microseconds(device->kernelWaitSleepDuration));
				waitKernelCount++;

				device->status = clGetEventInfo(device->kernelWaitEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, UINT32_LENGTH, &waitStatus, NULL);
				if (device->status != CL_SUCCESS)
				{
					onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting event info (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
					break;
				}
			}

			if (waitKernelCount > 15u) device->kernelWaitSleepDuration++; // hysteresis required to avoid constant changing of kernelWaitSleepDuration that will waste CPU cycles/hashrates
			else if (waitKernelCount < 5u && device->kernelWaitSleepDuration > 0u) device->kernelWaitSleepDuration--;
		}

		device->h_solutionCount = (uint32_t *)clEnqueueMapBuffer(device->queue, device->solutionCountBuffer, CL_TRUE, CL_MAP_READ, 0, UINT32_LENGTH, NULL, NULL, NULL, &device->status);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting solution count from device (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

		if (device->h_solutionCount[0] > 0u)
		{
			device->h_solutions = (uint64_t *)clEnqueueMapBuffer(device->queue, device->solutionsBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, UINT64_LENGTH * MAX_SOLUTION_COUNT_DEVICE, NULL, NULL, NULL, &device->status);
			if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting solutions from device (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

			std::set<uint64_t> uniqueSolutions;

			for (uint32_t i{ 0 }; i < MAX_SOLUTION_COUNT_DEVICE && i < device->h_solutionCount[0]; ++i)
			{
				uint64_t const tempSolution{ device->h_solutions[i] };
				if (tempSolution != 0u && uniqueSolutions.find(tempSolution) == uniqueSolutions.end())
					uniqueSolutions.emplace(tempSolution);
			}

			std::thread t{ &openCLSolver::submitSolutions, this, uniqueSolutions, std::string{ c_currentChallenge }, device->platformName, device->deviceEnum };
			t.detach();

			device->status = clEnqueueUnmapMemObject(device->queue, device->solutionsBuffer, device->h_solutions, NULL, NULL, NULL);
			if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error unmapping solutions from host (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

			device->status = clEnqueueUnmapMemObject(device->queue, device->solutionCountBuffer, device->h_solutionCount, NULL, NULL, NULL);
			if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error unmapping solution count from host (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

			device->h_solutionCount = (uint32_t *)clEnqueueMapBuffer(device->queue, device->solutionCountBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, UINT32_LENGTH, NULL, NULL, NULL, &device->status);
			if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting solution count from device (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

			device->h_solutionCount[0] = 0u;
		}

		device->status = clEnqueueUnmapMemObject(device->queue, device->solutionCountBuffer, device->h_solutionCount, NULL, NULL, NULL);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error unmapping solution count from host (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

	} while (device->mining);

	onMessage(device->platformName, device->deviceEnum, "Info", "Stop mining...");
	device->hashCount.store(0ull);

	clFinish(device->queue);

	clReleaseKernel(device->kernel);
	clReleaseProgram(device->program);
	clReleaseMemObject(device->solutionsBuffer);
	clReleaseMemObject(device->midstateBuffer);
	clReleaseCommandQueue(device->queue);
	clReleaseContext(device->context);

	device->initialized = false;
	onMessage(device->platformName, device->deviceEnum, "Info", "Mining stopped.");
}
