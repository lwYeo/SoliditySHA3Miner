#include "device.h"

namespace OpenCLSolver
{
	// --------------------------------------------------------------------
	// Static
	// --------------------------------------------------------------------

	std::vector<std::unique_ptr<Device>> Device::devices;
	const char *Device::kernelSource;
	const char *Device::kernelSourceKing;
	size_t Device::kernelSourceSize;
	size_t Device::kernelSourceKingSize;

	template<typename T>
	const char* Device::getOpenCLErrorCodeStr(T &input)
	{
		int errorCode = (int)input;
		switch (errorCode)
		{
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
			return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case CL_PLATFORM_NOT_FOUND_KHR:
			return "CL_PLATFORM_NOT_FOUND_KHR";
			//case CL_INVALID_PROPERTY_EXT:
			//    return "CL_INVALID_PROPERTY_EXT";
		case CL_DEVICE_PARTITION_FAILED_EXT:
			return "CL_DEVICE_PARTITION_FAILED_EXT";
		case CL_INVALID_PARTITION_COUNT_EXT:
			return "CL_INVALID_PARTITION_COUNT_EXT";
		case CL_INVALID_DEVICE_QUEUE:
			return "CL_INVALID_DEVICE_QUEUE";
		case CL_INVALID_PIPE_SIZE:
			return "CL_INVALID_PIPE_SIZE";
		default:
			return "unknown error code";
		}
	}

	void Device::preInitialize(std::string sha3Kernel, std::string sha3KingKernel)
	{
		kernelSourceSize = sha3Kernel.size();
		kernelSourceKingSize = sha3KingKernel.size();

		kernelSource = (char *)std::malloc(kernelSourceSize + 1);
		kernelSourceKing = (char *)std::malloc(kernelSourceKingSize + 1);

#	ifdef __linux__
		strcpy((char *)kernelSource, sha3Kernel.c_str());
		strcpy((char *)kernelSourceKing, sha3KingKernel.c_str());
#	else
		strcpy_s((char *)kernelSource, kernelSourceSize + 1, sha3Kernel.c_str());
		strcpy_s((char *)kernelSourceKing, kernelSourceKingSize + 1, sha3KingKernel.c_str());
#	endif
	}

	// --------------------------------------------------------------------
	// Public
	// --------------------------------------------------------------------

	Device::Device(int devEnum, cl_device_id devID, cl_device_type devType, cl_platform_id devPlatformID, bool isKingMaking,
		float const userDefIntensity, uint32_t userLocalWorkSize) :
		status{ CL_SUCCESS },
		computeCapability{ 0 },
		deviceEnum{ devEnum },
		deviceID{ devID },
		deviceType{ devType },
		hashCount{ 0ull },
		hashStartTime{ std::chrono::steady_clock::now() },
		initialized{ false },
		kernelWaitSleepDuration{ 1000u },
		mining{ false },
		platformID{ devPlatformID },
		userDefinedIntensity{ userDefIntensity },
		pciBusID{ 0 }
	{
		char charBuffer[1024];
		size_t sizeBuffer[3];

		status = clGetPlatformInfo(platformID, CL_PLATFORM_NAME, sizeof(charBuffer), charBuffer, NULL);
		platformName = (std::string{ charBuffer } == "") ? "Unknown" : charBuffer;
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_PLATFORM_NAME.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR, sizeof(charBuffer), charBuffer, NULL);
		vendor = (std::string{ charBuffer } == "") ? "Unknown" : charBuffer;
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_VENDOR.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, sizeof(charBuffer), charBuffer, NULL);
		name = (std::string{ charBuffer } == "") ? "Unknown" : charBuffer;
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_NAME.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof(charBuffer), charBuffer, NULL);
		openCLVersion = (std::string{ charBuffer } == "") ? "Unknown" : charBuffer;
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_OPENCL_C_VERSION.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, sizeof(charBuffer), charBuffer, NULL);
		extensions = charBuffer;
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_EXTENSIONS.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizeBuffer), sizeBuffer, NULL);
		maxWorkItemSizes.insert(maxWorkItemSizes.end(), sizeBuffer, sizeBuffer + sizeof(sizeBuffer) / sizeof(size_t));
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_MAX_WORK_ITEM_SIZES.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_MAX_WORK_GROUP_SIZE.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_MAX_COMPUTE_UNITS.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_MAX_MEM_ALLOC_SIZE.");

		status = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
		if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CL_DEVICE_GLOBAL_MEM_SIZE.");

		if (isCUDA())
		{
			cl_uint computeCapabilityMajor, computeCapabilityMinor;
			status = clGetDeviceInfo(deviceID, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &computeCapabilityMajor, NULL);
			if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CUDA compute capability.");

			status = clGetDeviceInfo(deviceID, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &computeCapabilityMinor, NULL);
			if (status != CL_SUCCESS) throw std::runtime_error("Failed to get CUDA compute capability.");

			computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
		}
		else if (isAPP())
		{
			cl_device_topology_amd topology;
			status = clGetDeviceInfo(deviceID, CL_DEVICE_TOPOLOGY_AMD, sizeof(cl_device_topology_amd), &topology, NULL);

			if (status == CL_SUCCESS)
				if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
				{
					pciBusID = (unsigned int)topology.pcie.bus;

					if (ADL_API::foundAdlApi())
					{
						m_api.assignPciBusID(pciBusID);

						std::string realName;
						m_api.getAdapterName(&realName);
						if (!realName.empty()) name = realName;
					}
				}
		}

		if (userLocalWorkSize > 0)
		{
			localWorkSize = (userLocalWorkSize > maxWorkGroupSize) ? maxWorkGroupSize : userLocalWorkSize;
			localWorkSize = (uint32_t)(localWorkSize / 64) * 64; // in multiples of 64
		}
		else if (isINTEL()) localWorkSize = 64; // iGPU

		else localWorkSize = DEFAULT_LOCAL_WORK_SIZE;

		setIntensity(userDefinedIntensity, isKingMaking);
	}

	bool Device::isAPP()
	{
		std::string tempPlatform{ platformName };
		std::transform(tempPlatform.begin(), tempPlatform.end(), tempPlatform.begin(), ::toupper);
		return tempPlatform.find("ACCELERATED PARALLEL PROCESSING") != std::string::npos;
	}

	bool Device::isCUDA()
	{
		std::string tempPlatform{ platformName };
		std::transform(tempPlatform.begin(), tempPlatform.end(), tempPlatform.begin(), ::toupper);
		return tempPlatform.find("CUDA") != std::string::npos;
	}

	bool Device::isINTEL()
	{
		std::string tempPlatform{ platformName };
		std::transform(tempPlatform.begin(), tempPlatform.end(), tempPlatform.begin(), ::toupper);
		return tempPlatform.find("INTEL") != std::string::npos;
	}

	std::string Device::getName()
	{
		return name;
	}

	bool Device::getSettingMaxCoreClock(int *maxCoreClock, std::string *errorMessage)
	{
		return m_api.getSettingMaxCoreClock(maxCoreClock, errorMessage);
	}

	bool Device::getSettingMaxMemoryClock(int *mamMemoryCloxk, std::string *errorMessage)
	{
		return m_api.getSettingMaxMemoryClock(mamMemoryCloxk, errorMessage);
	}

	bool Device::getSettingPowerLimit(int *powerLimit, std::string *errorMessage)
	{
		return m_api.getSettingPowerLimit(powerLimit, errorMessage);
	}

	bool Device::getSettingThermalLimit(int *thermalLimit, std::string *errorMessage)
	{
		return m_api.getSettingThermalLimit(thermalLimit, errorMessage);
	}

	bool Device::getSettingFanLevelPercent(int *fanLevel, std::string *errorMessage)
	{
		return m_api.getSettingFanLevelPercent(fanLevel, errorMessage);
	}

	bool Device::getCurrentFanTachometerRPM(int *tachometerRPM, std::string *errorMessage)
	{
		return m_api.getCurrentFanTachometerRPM(tachometerRPM, errorMessage);
	}

	bool Device::getCurrentTemperature(int *temperature, std::string *errorMessage)
	{
		return m_api.getCurrentTemperature(temperature, errorMessage);
	}

	bool Device::getCurrentCoreClock(int *coreClock, std::string *errorMessage)
	{
		return m_api.getCurrentCoreClock(coreClock, errorMessage);
	}

	bool Device::getCurrentMemoryClock(int *memoryClock, std::string *errorMessage)
	{
		return m_api.getCurrentMemoryClock(memoryClock, errorMessage);
	}

	bool Device::getCurrentUtilizationPercent(int *utilization, std::string *errorMessage)
	{
		return m_api.getCurrentUtilizationPercent(utilization, errorMessage);
	}

	uint64_t Device::hashRate()
	{
		using namespace std::chrono;
		return (uint64_t)((long double)hashCount.load() / (duration_cast<seconds>(steady_clock::now() - hashStartTime).count()));
	}

	bool Device::setKernelArgs(std::string& errorMessage, bool const isKingMaking)
	{
		errorMessage = "";

		if (isKingMaking)
		{
			status = clSetKernelArg(kernel, 0u, sizeof(cl_mem), &messageBuffer);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Error setting message buffer to kernel (" } +Device::getOpenCLErrorCodeStr(status) + ")...";
				return false;
			}
		}
		else
		{
			status = clSetKernelArg(kernel, 0u, sizeof(cl_mem), &midstateBuffer);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Error setting midsate buffer to kernel (" } +Device::getOpenCLErrorCodeStr(status) + ")...";
				return false;
			}
		}

		status = clSetKernelArg(kernel, 1u, sizeof(cl_mem), &targetBuffer);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Error setting target buffer to kernel (" } +Device::getOpenCLErrorCodeStr(status) + ")...";
			return false;
		}

		status = clSetKernelArg(kernel, 3u, sizeof(cl_mem), &solutionsBuffer);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Error setting solutions buffer to kernel (" } +Device::getOpenCLErrorCodeStr(status) + ")...";
			return false;
		}

		status = clSetKernelArg(kernel, 4u, sizeof(cl_mem), &solutionCountBuffer);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Error setting solution count buffer to kernel (" } +Device::getOpenCLErrorCodeStr(status) + ")...";
			return false;
		}

		return true;
	}

	void Device::initialize(std::string& errorMessage, bool const isKingMaking)
	{
		errorMessage = "";
		cl_context_properties contextProp[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0 };

		context = clCreateContext(contextProp, 1u, &deviceID, NULL, NULL, &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to create context (" } +getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		queue = clCreateCommandQueue(context, deviceID, 0, &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to create command queue (" } +getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		h_solutions = reinterpret_cast<uint64_t *>(malloc(UINT64_LENGTH * MAX_SOLUTION_COUNT_DEVICE));
		std::memset(h_solutions, 0u, UINT64_LENGTH * MAX_SOLUTION_COUNT_DEVICE);

		solutionsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, UINT64_LENGTH * MAX_SOLUTION_COUNT_DEVICE, h_solutions, &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to use solutions buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		h_solutionCount = reinterpret_cast<uint32_t *>(malloc(UINT32_LENGTH));
		std::memset(h_solutionCount, 0u, UINT32_LENGTH);

		solutionCountBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, UINT32_LENGTH, h_solutionCount, &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to use solution count buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		std::string newSource;
		std::string kernelEntryName;

		if (isKingMaking)
		{
			newSource = kernelSourceKing;
			kernelEntryName = "hashMessage";

			messageBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, MESSAGE_LENGTH, NULL, &status);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Failed to allocate message buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
				return;
			}

			targetBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, UINT256_LENGTH, NULL, &status);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Failed to allocate target buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
				return;
			}
		}
		else
		{
			newSource = kernelSource;
			kernelEntryName = "hashMidstate";

			midstateBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, SPONGE_LENGTH, NULL, &status);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Failed to allocate midstate buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
				return;
			}

			targetBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, UINT64_LENGTH, NULL, &status);
			if (status != CL_SUCCESS)
			{
				errorMessage = std::string{ "Failed to allocate target buffer (" } +Device::getOpenCLErrorCodeStr(status) + ')';
				return;
			}
		}

		if (isAPP())
		{
			newSource.insert(0, "#define PLATFORM 2\n");
		}
		else if (isCUDA())
		{
			newSource.insert(0, std::string{ "#define PLATFORM 1\n" } +
				"#define COMPUTE " + std::to_string(computeCapability) + "\n");
		}

		const char *tempSouce = newSource.c_str();
		size_t tempSize = newSource.size();

		program = clCreateProgramWithSource(context, 1u, &tempSouce, (const size_t *)&tempSize, &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to create program (" } +getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		status = clBuildProgram(program, 1u, &deviceID, NULL, NULL, NULL);
		if (status != CL_SUCCESS)
		{
			size_t log_size;
			clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			char *log = (char *)malloc(log_size);
			clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

			errorMessage = std::string{ "Failed to build program (" } +getOpenCLErrorCodeStr(status) + ")\n" + log;
			return;
		}

		kernel = clCreateKernel(program, kernelEntryName.c_str(), &status);
		if (status != CL_SUCCESS)
		{
			errorMessage = std::string{ "Failed to create kernel from program (" } +getOpenCLErrorCodeStr(status) + ')';
			return;
		}

		if (!setKernelArgs(errorMessage, isKingMaking)) return;;

		initialized = true;
	}

	void Device::setIntensity(float const intensity, bool isKingMaking)
	{
		if (isINTEL()) userDefinedIntensity = (intensity > 1.0f) ? intensity : 18.0f; // iGPU
		else userDefinedIntensity = (intensity > 1.0f) ? intensity : (isKingMaking ? DEFAULT_INTENSITY_KING : DEFAULT_INTENSITY);

		auto userTotalWorkSize = (uint32_t)std::pow(2, userDefinedIntensity);
		globalWorkSize = (uint32_t)(userTotalWorkSize / localWorkSize) * localWorkSize; // in multiples of localWorkSize
	}
}