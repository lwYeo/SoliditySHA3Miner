#pragma unmanaged

/*
Author: Mikers, Azlehria, lwYeo
Date: March 4 - July, 2018 for 0xbitcoin dev

Hashburner Enhancements for COSMiC: LtTofu (Mag517)
Date: April 24, 2018

based off of https://github.com/Dunhili/SHA3-gpu-brute-force-cracker/blob/master/sha3.cu

* Author: Brian Bowden
* Date: 5/12/14
*
* This is the parallel version of SHA-3.
*/

#include "cudaErrorCheck.cu"
#include "cudasolver.h"

#define MAX_SOLUTION_COUNT_DEVICE 32
__constant__ uint32_t maxSolutionCount = MAX_SOLUTION_COUNT_DEVICE;

__constant__ uint64_t d_midState[MIDSTATE_LENGTH];
__constant__ uint64_t d_target;

__device__ __constant__ uint64_t const RC[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__device__ __forceinline__ uint64_t ROTL64(uint64_t x, uint32_t y)
{
	return (x << y) ^ (x >> (64 - y));
}

__device__ __forceinline__ uint64_t ROTR64(uint64_t x, uint32_t y)
{
	return (x >> y) ^ (x << (64 - y));
}

__device__ __forceinline__ uint64_t bswap_64(uint64_t const input)
{
	uint64_t output;
	asm("{"
		"  prmt.b32 %0, %3, 0, 0x0123;"
		"  prmt.b32 %1, %2, 0, 0x0123;"
		"}" : "=r"(reinterpret_cast<uint2&>(output).x), "=r"(reinterpret_cast<uint2&>(output).y)
		: "r"(reinterpret_cast<uint2 const&>(input).x), "r"(reinterpret_cast<uint2 const&>(input).y));
	return output;
}

__device__ __forceinline__ uint32_t bswap_32(uint32_t const input)
{
	uint32_t output;
	asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(output) : "r"(input));
	return output;
}

__device__ __forceinline__ uint64_t xor5(uint64_t const a, uint64_t const b, uint64_t const c, uint64_t const d, uint64_t const e)
{
	uint64_t output;
	asm("{"
		"  xor.b64 %0, %1, %2;"
		"  xor.b64 %0, %0, %3;"
		"  xor.b64 %0, %0, %4;"
		"  xor.b64 %0, %0, %5;"
		"}" : "=l"(output) : "l"(a), "l"(b), "l"(c), "l"(d), "l"(e));
	return output;
}

__device__ __forceinline__ uint64_t xor3(uint64_t const a, uint64_t const b, uint64_t const c)
{
	uint64_t output;
#if __CUDA_ARCH__ >= 500
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0x96;"
		"  lop3.b32 %1, %3, %5, %7, 0x96;"
		"}" : "=r"(reinterpret_cast<uint2&>(output).x), "=r"(reinterpret_cast<uint2&>(output).y)
		: "r"(reinterpret_cast<uint2 const&>(a).x), "r"(reinterpret_cast<uint2 const&>(a).y),
		"r"(reinterpret_cast<uint2 const&>(b).x), "r"(reinterpret_cast<uint2 const&>(b).y),
		"r"(reinterpret_cast<uint2 const&>(c).x), "r"(reinterpret_cast<uint2 const&>(c).y));
#else
	asm("{"
		"  xor.b64 %0, %1, %2;"
		"  xor.b64 %0, %0, %3;"
		"}" : "=l"(output) : "l"(a), "l"(b), "l"(c));
#endif
	return output;
}

__device__ __forceinline__ uint64_t chi(uint64_t const a, uint64_t const b, uint64_t const c)
{
#if __CUDA_ARCH__ >= 500
	uint64_t output;
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0xD2;"
		"  lop3.b32 %1, %3, %5, %7, 0xD2;"
		"}" : "=r"(reinterpret_cast<uint2&>(output).x), "=r"(reinterpret_cast<uint2&>(output).y)
		: "r"(reinterpret_cast<uint2 const&>(a).x), "r"(reinterpret_cast<uint2 const&>(a).y),
		"r"(reinterpret_cast<uint2 const&>(b).x), "r"(reinterpret_cast<uint2 const&>(b).y),
		"r"(reinterpret_cast<uint2 const&>(c).x), "r"(reinterpret_cast<uint2 const&>(c).y));
	return output;
#else
	return a ^ ((~b) & c);
#endif
}

// shortcut to rotation by 32 (flip halves), then rotate left by `mag`
__device__ __forceinline__ uint64_t ROTLfrom32(uint64_t rtdby32, uint32_t magnitude)
{
	asm("{"
		"    .reg .b32 hi, lo, scr, mag;       "
		"    mov.b64 {lo,hi}, %0;              "      // halves reversed since rotl'd by 32
		"    mov.b32 mag, %1;                  "
		"    shf.l.wrap.b32 scr, lo, hi, mag;  "
		"    shf.l.wrap.b32 lo, hi, lo, mag;   "
		"    mov.b64 %0, {scr,lo};             "
		"}" : "+l"(rtdby32) : "r"(magnitude));    // see if this is faster w/ uint2 .x and .y
												  // for saving shf results out
	return rtdby32;   // return rotation from the rotation by 32
}

// shortcut to rotation by 32 (flip halves), then rotate right by `mag`
__device__ __forceinline__ uint64_t ROTRfrom32(uint64_t rtdby32, uint32_t magnitude)
{
	asm("{"
		"    .reg .b32 hi, lo, scr, mag;       "
		"    mov.b64 {lo,hi}, %0;              "      // halves reversed since rotl'd by 32
		"    mov.b32 mag, %1;                  "
		"    shf.r.wrap.b32 scr, hi, lo, mag;  "
		"    shf.r.wrap.b32 lo, lo, hi, mag;   "
		"    mov.b64 %0, {scr,lo};             "
		"}" : "+l"(rtdby32) : "r"(magnitude));    // see if this is faster w/ uint2 .x and .y
												  // for saving shf results out
	return rtdby32;   // return rotation from the rotation by 32
}

__global__ void cuda_mine(uint64_t* __restrict__ solutions, uint32_t* __restrict__ solution_count, uint64_t const threads)
{
	uint64_t const nounce{ threads + (blockDim.x * blockIdx.x + threadIdx.x) };

	uint64_t state[25], C[5], D[5];
	uint64_t n[11]{ ROTL64(nounce, 7) };
	n[1] = ROTL64(n[0], 1);
	n[2] = ROTL64(n[1], 6);
	n[3] = ROTL64(n[2], 2);
	n[4] = ROTL64(n[3], 4);
	n[5] = ROTL64(n[4], 7);
	n[6] = ROTL64(n[5], 12);
	n[7] = ROTL64(n[6], 5);
	n[8] = ROTL64(n[7], 11);
	n[9] = ROTL64(n[8], 7);
	n[10] = ROTL64(n[9], 1);

	C[0] = d_midState[0];
	C[1] = d_midState[1];
	C[2] = d_midState[2] ^ n[7];
	C[3] = d_midState[3];
	C[4] = d_midState[4] ^ n[2];
	state[0] = chi(C[0], C[1], C[2]) ^ RC[0];
	state[1] = chi(C[1], C[2], C[3]);
	state[2] = chi(C[2], C[3], C[4]);
	state[3] = chi(C[3], C[4], C[0]);
	state[4] = chi(C[4], C[0], C[1]);

	C[0] = d_midState[5];
	C[1] = d_midState[6] ^ n[4];
	C[2] = d_midState[7];
	C[3] = d_midState[8];
	C[4] = d_midState[9] ^ n[9];
	state[5] = chi(C[0], C[1], C[2]);
	state[6] = chi(C[1], C[2], C[3]);
	state[7] = chi(C[2], C[3], C[4]);
	state[8] = chi(C[3], C[4], C[0]);
	state[9] = chi(C[4], C[0], C[1]);

	C[0] = d_midState[10];
	C[1] = d_midState[11] ^ n[0];
	C[2] = d_midState[12];
	C[3] = d_midState[13] ^ n[1];
	C[4] = d_midState[14];
	state[10] = chi(C[0], C[1], C[2]);
	state[11] = chi(C[1], C[2], C[3]);
	state[12] = chi(C[2], C[3], C[4]);
	state[13] = chi(C[3], C[4], C[0]);
	state[14] = chi(C[4], C[0], C[1]);

	C[0] = d_midState[15] ^ n[5];
	C[1] = d_midState[16];
	C[2] = d_midState[17];
	C[3] = d_midState[18] ^ n[3];
	C[4] = d_midState[19];
	state[15] = chi(C[0], C[1], C[2]);
	state[16] = chi(C[1], C[2], C[3]);
	state[17] = chi(C[2], C[3], C[4]);
	state[18] = chi(C[3], C[4], C[0]);
	state[19] = chi(C[4], C[0], C[1]);

	C[0] = d_midState[20] ^ n[10];
	C[1] = d_midState[21] ^ n[8];
	C[2] = d_midState[22] ^ n[6];
	C[3] = d_midState[23];
	C[4] = d_midState[24];
	state[20] = chi(C[0], C[1], C[2]);
	state[21] = chi(C[1], C[2], C[3]);
	state[22] = chi(C[2], C[3], C[4]);
	state[23] = chi(C[3], C[4], C[0]);
	state[24] = chi(C[4], C[0], C[1]);

#if __CUDA_ARCH__ >= 350
#  pragma unroll
#endif
	for (uint_fast8_t i{ 1 }; i < 23; ++i)
	{
		for (uint_fast8_t x{ 0 }; x < 5; ++x)
			C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20]);

#if __CUDA_ARCH__ >= 350
		for (uint_fast8_t x{ 0 }; x < 5; ++x)
		{
			D[x] = ROTL64(C[(x + 2) % 5], 1);
			state[x] = xor3(state[x], D[x], C[x]);
			state[x + 5] = xor3(state[x + 5], D[x], C[x]);
			state[x + 10] = xor3(state[x + 10], D[x], C[x]);
			state[x + 15] = xor3(state[x + 15], D[x], C[x]);
			state[x + 20] = xor3(state[x + 20], D[x], C[x]);
		}
#else
		for (uint_fast8_t x{ 0 }; x < 5; ++x)
		{
			D[x] = ROTL64(C[(x + 2) % 5], 1) ^ C[x];
			state[x] = state[x] ^ D[x];
			state[x + 5] = state[x + 5] ^ D[x];
			state[x + 10] = state[x + 10] ^ D[x];
			state[x + 15] = state[x + 15] ^ D[x];
			state[x + 20] = state[x + 20] ^ D[x];
		}
#endif

		C[0] = state[1];
		state[1] = ROTR64(state[6], 20);
		state[6] = ROTL64(state[9], 20);
		state[9] = ROTR64(state[22], 3);
		state[22] = ROTR64(state[14], 25);
		state[14] = ROTL64(state[20], 18);
		state[20] = ROTR64(state[2], 2);
		state[2] = ROTR64(state[12], 21);
		state[12] = ROTL64(state[13], 25);
		state[13] = ROTL64(state[19], 8);
		state[19] = ROTR64(state[23], 8);
		state[23] = ROTR64(state[15], 23);

#if __CUDA_ARCH__ >= 320
		state[15] = ROTRfrom32(state[4], 5);
#else
		state[15] = ROTL64(state[4], 27);
#endif

		state[4] = ROTL64(state[24], 14);
		state[24] = ROTL64(state[21], 2);
		state[21] = ROTR64(state[8], 9);
		state[8] = ROTR64(state[16], 19);

#if __CUDA_ARCH__ >= 320
		state[16] = ROTLfrom32(state[5], 4);
		state[5] = ROTRfrom32(state[3], 4);
#else
		state[16] = ROTR64(state[5], 28);
		state[5] = ROTL64(state[3], 28);
#endif

		state[3] = ROTL64(state[18], 21);
		state[18] = ROTL64(state[17], 15);
		state[17] = ROTL64(state[11], 10);
		state[11] = ROTL64(state[7], 6);
		state[7] = ROTL64(state[10], 3);
		state[10] = ROTL64(C[0], 1);

		for (uint_fast8_t x{ 0 }; x < 25; x += 5)
		{
			C[0] = state[x];
			C[1] = state[x + 1];
			C[2] = state[x + 2];
			C[3] = state[x + 3];
			C[4] = state[x + 4];
			state[x] = chi(C[0], C[1], C[2]);
			state[x + 1] = chi(C[1], C[2], C[3]);
			state[x + 2] = chi(C[2], C[3], C[4]);
			state[x + 3] = chi(C[3], C[4], C[0]);
			state[x + 4] = chi(C[4], C[0], C[1]);
		}

		state[0] = state[0] ^ RC[i];
	}

#if __CUDA_ARCH__ >= 350
#  pragma unroll
#endif
	for (uint_fast8_t x{ 0 }; x < 5; ++x)
		C[(x + 6) % 5] = xor5(state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20]);

	D[0] = ROTL64(C[2], 1);
	D[1] = ROTL64(C[3], 1);
	D[2] = ROTL64(C[4], 1);

	state[0] = xor3(state[0], D[0], C[0]);
	state[6] = xor3(state[6], D[1], C[1]);
	state[12] = xor3(state[12], D[2], C[2]);
	state[6] = ROTR64(state[6], 20);
	state[12] = ROTR64(state[12], 21);

	state[0] = chi(state[0], state[6], state[12]) ^ RC[23];

	if (bswap_64(state[0]) < d_target)
	{
		(*solution_count)++;
		if ((*solution_count) < maxSolutionCount) solutions[(*solution_count) - 1] = nounce;
	}
}

// --------------------------------------------------------------------
// CUDASolver
// --------------------------------------------------------------------

void CUDASolver::pushTarget()
{
	std::string const tgtPrefix(static_cast<std::string::size_type>(UINT256_LENGTH * 2) - m_target.GetHex().length(), '0');

	uint64_t truncTarget{ std::stoull((tgtPrefix + s_target.substr(2)).substr(0, 16), nullptr, 16) };

	for (auto& device : m_devices)
	{
		cudaSetDevice(device->deviceID);

		cudaMemcpyToSymbol(d_target, &truncTarget, UINT64_LENGTH, 0, cudaMemcpyHostToDevice);
	}
	m_newTarget.store(false);
}

void CUDASolver::pushMessage()
{
	for (auto& device : m_devices)
	{
		cudaSetDevice(device->deviceID);

		cudaMemcpyToSymbol(d_midState, getMidState(m_miningMessage).data(), STATE_LENGTH, 0, cudaMemcpyHostToDevice);
	}
	m_newMessage.store(false);
}

void CUDASolver::checkInputs(std::unique_ptr<Device>& device, char *currentChallenge)
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
		m_solutionHashCount.store(0ull);
		m_solutionHashStartTime.store(std::chrono::steady_clock::now());

		if (m_newTarget.load()) pushTarget();

		if (m_newMessage.load())
		{
			strcpy_s(currentChallenge, s_challenge.size() + 1, s_challenge.c_str());

			std::memcpy(&m_miningMessage, &m_prefix, PREFIX_LENGTH);
			std::memcpy(&m_miningMessage[PREFIX_LENGTH], &m_solutionTemplate, UINT256_LENGTH);
			pushMessage();
		}

		cudaSetDevice(device->deviceID); // required to set back after push target/messsage
	}
}

void CUDASolver::findSolution(int const deviceID)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device) { return device->deviceID == deviceID; });

	CudaSafeCall(cudaSetDevice(device->deviceID));

	if (!device->initialized)
	{
		onMessage(device->deviceID, "Info", "Initializing device...");

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
	}

	uint64_t currentSearchSpace = UINT64_MAX;
	char *c_currentChallenge = (char *)malloc(s_challenge.size());
	strcpy_s(c_currentChallenge, s_challenge.size() + 1, s_challenge.c_str());

	onMessage(device->deviceID, "Info", "Start mining...");
	onMessage(device->deviceID, "Debug", "Threads: " + std::to_string(device->threads()) + " Grid size: " + std::to_string(device->grid().x) + " Block size:" + std::to_string(device->block().x));

	device->mining = true;
	device->hashCount.store(0ull);
	device->hashStartTime.store(std::chrono::steady_clock::now());
	do
	{
		while (m_pause.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

		checkInputs(device, c_currentChallenge);

		if (currentSearchSpace == UINT64_MAX) currentSearchSpace = getNextSearchSpace(device);

		cuda_mine<<<device->grid(), device->block()>>>(device->d_Solutions, device->d_SolutionCount, currentSearchSpace);

		CudaCheckError();

		cudaError_t response = cudaDeviceSynchronize();
		if (response == cudaSuccess) currentSearchSpace = UINT64_MAX;
		else
		{
			std::string cudaErrors;
			cudaError_t lastError = cudaGetLastError();
			while (lastError != cudaSuccess)
			{
				if (!cudaErrors.empty()) cudaErrors += " <- ";
				cudaErrors += cudaGetErrorString(lastError);
				lastError = cudaGetLastError();
			}
			onMessage(device->deviceID, "Error", "Kernel launch failed: " + cudaErrors);

			device->mining = false;
			break;
		}

		if (*device->h_SolutionCount > 0u)
		{
			std::set<uint64_t> uniqueSolutions;

			for (uint32_t i{ 0u }; i < MAX_SOLUTION_COUNT_DEVICE && i < *device->h_SolutionCount; i++)
			{
				uint64_t const tempSolution{ device->h_Solutions[i] };

				if (tempSolution != 0u && uniqueSolutions.find(tempSolution) == uniqueSolutions.end())
					uniqueSolutions.emplace(tempSolution);
			}

			std::thread t{ &CUDASolver::submitSolutions, this, uniqueSolutions, std::string{ c_currentChallenge } };
			t.detach();

			std::memset(device->h_SolutionCount, 0u, UINT32_LENGTH);
		}
	} while (device->mining);

	onMessage(device->deviceID, "Info", "Stop mining...");
	device->hashCount.store(0ull);

	CudaSafeCall(cudaSetDevice(device->deviceID));

	CudaSafeCall(cudaFreeHost(device->h_SolutionCount));
	CudaSafeCall(cudaFreeHost(device->h_Solutions));
	CudaSafeCall(cudaDeviceReset());

	device->initialized = false;
	onMessage(device->deviceID, "Info", "Mining stopped.");
}
