#pragma unmanaged
#include "openCLSolver.h"

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

std::vector<openCLSolver::Platform> openCLSolver::platforms;
std::atomic<bool> openCLSolver::m_newTarget;
std::atomic<bool> openCLSolver::m_newMessage;
std::atomic<bool> openCLSolver::m_pause;

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

openCLSolver::openCLSolver(std::string const maxDifficulty, std::string solutionTemplate, std::string kingAddress) noexcept :
	s_address{ "" },
	s_challenge{ "" },
	s_target{ "" },
	s_difficulty{ "" },
	m_address{ 0 },
	m_solutionTemplate{ 0 },
	m_prefix{ 0 },
	m_miningMessage{ 0 },
	m_target{ 0 },
	m_difficulty{ 0 },
	m_maxDifficulty{ maxDifficulty },
	s_kingAddress{ kingAddress }
{
	m_newTarget.store(false);
	m_newMessage.store(false);
	m_pause.store(false);

	m_solutionHashStartTime.store(std::chrono::steady_clock::now());

	hexStringToBytes(solutionTemplate, m_solutionTemplate);
}

openCLSolver::~openCLSolver() noexcept
{
	stopFinding();
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
	return openCLSolver::m_pause.load();
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

				std::string initMessage{ "Assigned OpenCL device (" + m_devices.back()->name + ")..." };
				initMessage += "\n Intensity: " + std::to_string(m_devices.back()->userDefinedIntensity);
				onMessage(platformName.c_str(), deviceEnum, "Info", initMessage);
				return true;
			}
			catch (std::exception ex)
			{
				onMessage(platformName.c_str(), deviceEnum, "Error", ex.what());
				return false;
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

	m_newMessage.store(true);
	onMessage("", -1, "Info", "New challenge detected " + s_challenge.substr(0, 18) + "...");
}

void openCLSolver::updateTarget(std::string const target)
{
	if (m_customDifficulty > 0u && !(target == (m_maxDifficulty / m_customDifficulty).GetHex())) return;

	arith_uint256 tempTarget = arith_uint256(target);
	if (tempTarget == m_target) return;

	s_target = (target.substr(0, 2) == "0x") ? target : "0x" + target;
	m_target = tempTarget;

	m_newTarget.store(true);
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

void openCLSolver::startFinding()
{
	using namespace std::chrono_literals;

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
	using namespace std::chrono_literals;

	for (auto& device : m_devices) device->mining = false;
	std::this_thread::sleep_for(1s);

	m_newMessage.store(false);
	m_newTarget.store(false);
}

void openCLSolver::pauseFinding(bool pauseFinding)
{
	m_pause.store(pauseFinding);
}

// --------------------------------------------------------------------
// Private
// --------------------------------------------------------------------

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

void openCLSolver::onSolution(byte32_t const solution, std::string challenge)
{
	if (!isSubmitStale && challenge != s_challenge)
		return;
	else if (isSubmitStale && challenge != s_challenge)
		onMessage("", -1, "Warn", "Found stale solution, verifying...");
	else
		onMessage("", -1, "Info", "Found solution, verifying...");

	prefix_t prefix;
	std::memcpy(&prefix, &m_miningMessage, PREFIX_LENGTH);

	byte32_t emptySolution;
	std::memset(&emptySolution, 0u, UINT256_LENGTH);
	if (solution == emptySolution)
	{
		onMessage("", -1, "Error", "Verification failed: empty solution"
			+ std::string("\nChallenge: ") + challenge
			+ "\nAddress: " + s_address
			+ "\nTarget: " + s_target);
		return;
	}

	std::string prefixStr{ bytesToHexString(prefix) };
	std::string solutionStr{ bytesToHexString(solution) };

	std::string digestStr = keccak256(prefixStr + solutionStr);
	arith_uint256 digest = arith_uint256(digestStr);
	onMessage("", -1, "Debug", "Digest: 0x" + digestStr);

	if (digest >= m_target)
	{
		onMessage("", -1, "Error", "Verification failed: invalid solution"
			+ std::string("\nChallenge: ") + challenge
			+ "\nAddress: " + s_address
			+ "\nSolution: 0x" + solutionStr
			+ "\nDigest: 0x" + digestStr
			+ "\nTarget: " + s_target);
	}
	else
	{
		onMessage("", -1, "Info", "Solution verified, submitting nonce 0x" + solutionStr + "...");
		m_solutionCallback(("0x" + digestStr).c_str(), s_address.c_str(), challenge.c_str(), s_difficulty.c_str(), s_target.c_str(), ("0x" + solutionStr).c_str(), m_customDifficulty > 0u);
	}
}

void openCLSolver::submitSolutions(std::set<uint64_t> solutions, std::string challenge)
{
	for (uint64_t midStateSolution : solutions)
	{
		byte32_t solution{ m_solutionTemplate };
		if (s_kingAddress.empty())
			std::memcpy(&solution[12], &midStateSolution, UINT64_LENGTH); // keep first and last 12 bytes, fill middle 8 bytes for mid state
		else
			std::memcpy(&solution[ADDRESS_LENGTH], &midStateSolution, UINT64_LENGTH); // Shifted for King address

		onSolution(solution, challenge);
	}
}

const state_t openCLSolver::getMidState(message_t &newMessage)
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

uint64_t openCLSolver::getNextWorkPosition(std::unique_ptr<Device>& device)
{
	std::lock_guard<std::mutex> lock(m_searchSpaceMutex);

	device->hashCount += device->globalWorkSize;
	uint64_t lastPosition;
	incrementWorkPosition(lastPosition, device->globalWorkSize);
	return lastPosition;
}

void openCLSolver::pushTarget()
{
	std::string const tgtPrefix(static_cast<std::string::size_type>(UINT256_LENGTH * 2) - m_target.GetHex().length(), '0');

	uint64_t truncTarget{ std::stoull((tgtPrefix + s_target.substr(2)).substr(0, 16), nullptr, 16) };

	for (auto& device : m_devices)
	{
		if (!device->initialized) continue;

		device->status = clSetKernelArg(device->kernel, 2u, UINT64_LENGTH, &truncTarget);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting target buffer to kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
	}
	m_newTarget.store(false);
}

void openCLSolver::pushMessage()
{
	for (auto& device : m_devices)
	{
		if (!device->initialized) continue;

		device->status = clEnqueueWriteBuffer(device->queue, device->midstateBuffer, CL_TRUE, 0, STATE_LENGTH, getMidState(m_miningMessage).data(), NULL, NULL, NULL);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error writing to midstate buffer (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
	}
	m_newMessage.store(false);
}

void openCLSolver::checkInputs(std::unique_ptr<Device>& device, char *currentChallenge)
{
	std::lock_guard<std::mutex> lock(m_checkInputsMutex);

	if (m_newTarget.load() || m_newMessage.load())
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

		if (m_newTarget.load()) pushTarget();

		if (m_newMessage.load())
		{
			strcpy_s(currentChallenge, s_challenge.size() + 1, s_challenge.c_str());

			std::memcpy(&m_miningMessage, &m_prefix, PREFIX_LENGTH);
			std::memcpy(&m_miningMessage[PREFIX_LENGTH], &m_solutionTemplate, UINT256_LENGTH);
			pushMessage();
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

	device->h_Solutions = reinterpret_cast<uint64_t *>(malloc((MAX_SOLUTION_COUNT_DEVICE + 1) * UINT64_LENGTH));
	std::memset(device->h_Solutions, 0u, (MAX_SOLUTION_COUNT_DEVICE + 1) * UINT64_LENGTH);

	uint64_t currentWorkPosition = UINT64_MAX;
	char *c_currentChallenge = (char *)malloc(s_challenge.size());
	strcpy_s(c_currentChallenge, s_challenge.size() + 1, s_challenge.c_str());

	onMessage(device->platformName, device->deviceEnum, "Info", "Start mining...");
	onMessage(device->platformName, device->deviceEnum, "Debug", "Threads: " + std::to_string(device->globalWorkSize) + " Local work size: " + std::to_string(device->localWorkSize) + " Block size:" + std::to_string(device->globalWorkSize / device->localWorkSize));

	device->status = clSetKernelArg(device->kernel, 1u, sizeof(cl_mem), &device->solutionsBuffer);
	if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting solutions buffer to kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

	device->status = clSetKernelArg(device->kernel, 0u, sizeof(cl_mem), &device->midstateBuffer);
	if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting midsate buffer to kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

	device->mining = true;
	device->hashCount.store(0ull);
	device->hashStartTime.store(std::chrono::steady_clock::now());
	do
	{
		while (m_pause.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

		checkInputs(device, c_currentChallenge);

		if (currentWorkPosition == UINT64_MAX) currentWorkPosition = getNextWorkPosition(device);
		
		device->status = clSetKernelArg(device->kernel, 3u, UINT64_LENGTH, (void *)&currentWorkPosition);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error setting work positon buffer to kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

		device->status = clEnqueueNDRangeKernel(device->queue, device->kernel, 1u, NULL, &device->globalWorkSize, &device->localWorkSize, NULL, NULL, &device->kernelWaitEvent);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error starting kernel (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

		device->status = clFlush(device->queue);
		if (device->status != CL_SUCCESS)
		{
			onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error flushing commands (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
			if (device->status == CL_INVALID_COMMAND_QUEUE) // likely OC or intensity too high errors
			{
				using namespace std::chrono_literals;

				device->setIntensity(device->userDefinedIntensity - 0.1F);
				onMessage(device->platformName, device->deviceEnum, "Info", "Reducing intensity to " + std::to_string(device->userDefinedIntensity) + " in 10 seconds...");
				std::this_thread::sleep_for(10s);
				continue;
			}
		}
		else
		{
			//if (false) // Spinning in CUDA OpenCL
			//{
			//	device->status = clFinish(device->queue);
			//	if (device->status == CL_SUCCESS) currentWorkPosition = UINT64_MAX;
			//	else
			//	{
			//		onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting event info after flush (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
			//		break;
			//	}
			//}
			//else
			//{
				uint32_t waitStatus{ CL_QUEUED }, waitKernelCount{ 0 };
				while (waitStatus != CL_COMPLETE)
				{
					std::this_thread::sleep_for(std::chrono::microseconds(device->kernelWaitSleepDuration));
					waitKernelCount++;

					device->status = clGetEventInfo(device->kernelWaitEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, UINT32_LENGTH, &waitStatus, NULL);
					if (device->status != CL_SUCCESS)
					{
						onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting event info after flush (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
						break;
					}
				}
				currentWorkPosition = UINT64_MAX;
				if (waitKernelCount > 15u) device->kernelWaitSleepDuration++; // hysteresis required to avoid constant changing of kernelWaitSleepDuration that will waste CPU cycles/hashrates
				else if (waitKernelCount < 5u && device->kernelWaitSleepDuration > 0) device->kernelWaitSleepDuration--;
			//}
		}

		device->h_Solutions = (uint64_t *)clEnqueueMapBuffer(device->queue, device->solutionsBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, UINT64_LENGTH * (MAX_SOLUTION_COUNT_DEVICE + 1), NULL, NULL, NULL, &device->status);
		if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error getting solutions from device (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");

		if (device->h_Solutions[0] > 0u)
		{
			solution_t solutionCount;
			solutionCount.ulong_t = device->h_Solutions[0];

			std::set<uint64_t> uniqueSolutions;

			for (int32_t i{ 1 }; i < (MAX_SOLUTION_COUNT_DEVICE + 1) && i < solutionCount.int_t; ++i)
			{
				uint64_t const tempSolution{ device->h_Solutions[i] };
				if (tempSolution != 0u && uniqueSolutions.find(tempSolution) == uniqueSolutions.end())
					uniqueSolutions.emplace(tempSolution);
			}

			std::thread t{ &openCLSolver::submitSolutions, this, uniqueSolutions, std::string{ c_currentChallenge } };
			t.detach();

			std::memset(device->h_Solutions, 0u, UINT64_LENGTH * (MAX_SOLUTION_COUNT_DEVICE + 1));

			device->status = clEnqueueUnmapMemObject(device->queue, device->solutionsBuffer, device->h_Solutions, NULL, NULL, NULL);
			if (device->status != CL_SUCCESS) onMessage(device->platformName, device->deviceEnum, "Error", std::string{ "Error unmapping solutions from host (" } + Device::getOpenCLErrorCodeStr(device->status) + ")...");
		}
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
