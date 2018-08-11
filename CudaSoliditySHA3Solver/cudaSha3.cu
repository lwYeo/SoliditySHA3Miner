#pragma unmanaged

/*
Author: Mikers, Azlehria, optimized by lwYeo
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

typedef union
{
	uint2		uint2;
	uint64_t	uint64;
} nonce_t;

#define MAX_SOLUTION_COUNT_DEVICE 32

__constant__ uint64_t d_midState[MIDSTATE_LENGTH];

__constant__ uint64_t d_target[1];

__device__ __constant__ uint64_t const Keccak_f1600_RC[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__device__ __constant__ uint_fast8_t const MOD5[16] = { 
	0u, 1u, 2u, 3u, 4u,
	0u, 1u, 2u, 3u, 4u,
	0u, 1u, 2u, 3u, 4u,
	0u };

__device__ __forceinline__ nonce_t ROTL64(nonce_t const input, uint32_t const offset)
{
	nonce_t output;
	output.uint64 = (input.uint64 << offset) ^ (input.uint64 >> (64u - offset));
	return output;
}

__device__ __forceinline__ nonce_t ROTR64(nonce_t const input, uint32_t const offset)
{
	nonce_t output;
	output.uint64 = (input.uint64 >> offset) ^ (input.uint64 << (64u - offset));
	return output;
}

__device__ __forceinline__ nonce_t bswap_64(nonce_t const input)
{
	nonce_t output;
	asm("{"
		"  prmt.b32 %0, %3, 0, 0x0123;"
		"  prmt.b32 %1, %2, 0, 0x0123;"
		"}" : "=r"(output.uint2.x), "=r"(output.uint2.y) : "r"(input.uint2.x), "r"(input.uint2.y));
	return output;
}

__device__ __forceinline__ nonce_t xor5(nonce_t const a, nonce_t const b, nonce_t const c, nonce_t const d, nonce_t const e)
{
	nonce_t output;
	asm("{"
		"  xor.b64 %0, %1, %2;"
		"  xor.b64 %0, %0, %3;"
		"  xor.b64 %0, %0, %4;"
		"  xor.b64 %0, %0, %5;"
		"}" : "=l"(output.uint64) : "l"(a.uint64), "l"(b.uint64), "l"(c.uint64), "l"(d.uint64), "l"(e.uint64));
	return output;
}

__device__ __forceinline__ nonce_t xor3(nonce_t const a, nonce_t const b, nonce_t const c)
{
	nonce_t output;
#if __CUDA_ARCH__ >= 500
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0x96;"
		"  lop3.b32 %1, %3, %5, %7, 0x96;"
		"}" : "=r"(output.uint2.x), "=r"(output.uint2.y)
		: "r"(a.uint2.x), "r"(a.uint2.y), "r"(b.uint2.x), "r"(b.uint2.y), "r"(c.uint2.x), "r"(c.uint2.y));
#else
	asm("{"
		"  xor.b64 %0, %1, %2;"
		"  xor.b64 %0, %0, %3;"
		"}" : "=l"(output.uint64) : "l"(a.uint64), "l"(b.uint64), "l"(c.uint64));
#endif
	return output;
}

__device__ __forceinline__ nonce_t chi(nonce_t const a, nonce_t const b, nonce_t const c)
{
	nonce_t output;
#if __CUDA_ARCH__ >= 500
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0xD2;"
		"  lop3.b32 %1, %3, %5, %7, 0xD2;"
		"}" : "=r"(output.uint2.x), "=r"(output.uint2.y)
		: "r"(a.uint2.x), "r"(a.uint2.y), "r"(b.uint2.x), "r"(b.uint2.y), "r"(c.uint2.x), "r"(c.uint2.y));
#else
	output.uint64 = a.uint64 ^ ((~b.uint64) & c.uint64);
#endif
	return output;
}

// shortcut to rotation by 32 (flip halves), then rotate left by `mag`
__device__ __forceinline__ nonce_t ROTLfrom32(nonce_t rtdby32, uint32_t const magnitude)
{
	asm("{"
		"    .reg .b32 hi, lo, scr, mag;"
		"    mov.b64 {lo,hi}, %0;"						// halves reversed since rotl'd by 32
		"    mov.b32 mag, %1;"
		"    shf.l.wrap.b32 scr, lo, hi, mag;"
		"    shf.l.wrap.b32 lo, hi, lo, mag;"
		"    mov.b64 %0, {scr,lo};"
		"}" : "+l"(rtdby32.uint64) : "r"(magnitude));	// see if this is faster w/ uint2 .x and .y for saving shf results out
	return rtdby32;										// return rotation from the rotation by 32
}

// shortcut to rotation by 32 (flip halves), then rotate right by `mag`
__device__ __forceinline__ nonce_t ROTRfrom32(nonce_t rtdby32, uint32_t const magnitude)
{
	asm("{"
		"    .reg .b32 hi, lo, scr, mag;"
		"    mov.b64 {lo,hi}, %0;"						// halves reversed since rotl'd by 32
		"    mov.b32 mag, %1;"
		"    shf.r.wrap.b32 scr, hi, lo, mag;"
		"    shf.r.wrap.b32 lo, lo, hi, mag;"
		"    mov.b64 %0, {scr,lo};"
		"}" : "+l"(rtdby32.uint64) : "r"(magnitude));	// see if this is faster w/ uint2 .x and .y for saving shf results out
	return rtdby32;										// return rotation from the rotation by 32
}

__global__ void hashMidstate(uint64_t* __restrict__ solutions, uint32_t* __restrict__ solutionCount, uint64_t const startPosition)
{
	nonce_t nonce, state[25], C[5], D[5], n[11];
	nonce.uint64 = startPosition + (blockDim.x * blockIdx.x + threadIdx.x);

	n[0] = ROTL64(nonce, 7);
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

	C[0].uint64 = d_midState[0];
	C[1].uint64 = d_midState[1];
	C[2].uint64 = d_midState[2] ^ n[7].uint64;
	C[3].uint64 = d_midState[3];
	C[4].uint64 = d_midState[4] ^ n[2].uint64;
	state[0].uint64 = chi(C[0], C[1], C[2]).uint64 ^ Keccak_f1600_RC[0];
	state[1] = chi(C[1], C[2], C[3]);
	state[2] = chi(C[2], C[3], C[4]);
	state[3] = chi(C[3], C[4], C[0]);
	state[4] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midState[5];
	C[1].uint64 = d_midState[6] ^ n[4].uint64;
	C[2].uint64 = d_midState[7];
	C[3].uint64 = d_midState[8];
	C[4].uint64 = d_midState[9] ^ n[9].uint64;
	state[5] = chi(C[0], C[1], C[2]);
	state[6] = chi(C[1], C[2], C[3]);
	state[7] = chi(C[2], C[3], C[4]);
	state[8] = chi(C[3], C[4], C[0]);
	state[9] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midState[10];
	C[1].uint64 = d_midState[11] ^ n[0].uint64;
	C[2].uint64 = d_midState[12];
	C[3].uint64 = d_midState[13] ^ n[1].uint64;
	C[4].uint64 = d_midState[14];
	state[10] = chi(C[0], C[1], C[2]);
	state[11] = chi(C[1], C[2], C[3]);
	state[12] = chi(C[2], C[3], C[4]);
	state[13] = chi(C[3], C[4], C[0]);
	state[14] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midState[15] ^ n[5].uint64;
	C[1].uint64 = d_midState[16];
	C[2].uint64 = d_midState[17];
	C[3].uint64 = d_midState[18] ^ n[3].uint64;
	C[4].uint64 = d_midState[19];
	state[15] = chi(C[0], C[1], C[2]);
	state[16] = chi(C[1], C[2], C[3]);
	state[17] = chi(C[2], C[3], C[4]);
	state[18] = chi(C[3], C[4], C[0]);
	state[19] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midState[20] ^ n[10].uint64;
	C[1].uint64 = d_midState[21] ^ n[8].uint64;
	C[2].uint64 = d_midState[22] ^ n[6].uint64;
	C[3].uint64 = d_midState[23];
	C[4].uint64 = d_midState[24];
	state[20] = chi(C[0], C[1], C[2]);
	state[21] = chi(C[1], C[2], C[3]);
	state[22] = chi(C[2], C[3], C[4]);
	state[23] = chi(C[3], C[4], C[0]);
	state[24] = chi(C[4], C[0], C[1]);

#if __CUDA_ARCH__ >= 350
#  pragma unroll
#endif
	for (uint_fast8_t i{ 1u }; i < 23u; ++i)
	{
		C[1] = xor5(state[0], state[5], state[10], state[15], state[20]);
		C[2] = xor5(state[1], state[6], state[11], state[16], state[21]);
		C[3] = xor5(state[2], state[7], state[12], state[17], state[22]);
		C[4] = xor5(state[3], state[8], state[13], state[18], state[23]);
		C[0] = xor5(state[4], state[9], state[14], state[19], state[24]);

#if __CUDA_ARCH__ >= 350
		for (uint_fast8_t x{ 0u }; x < 5u; ++x)
		{
			D[x] = ROTL64(C[MOD5[x + 2]], 1);
			state[x] = xor3(state[x], D[x], C[x]);
			state[x + 5] = xor3(state[x + 5], D[x], C[x]);
			state[x + 10] = xor3(state[x + 10], D[x], C[x]);
			state[x + 15] = xor3(state[x + 15], D[x], C[x]);
			state[x + 20] = xor3(state[x + 20], D[x], C[x]);
	}
#else
		for (uint_fast8_t x{ 0u }; x < 5u; ++x)
		{
			D[x].uint64 = ROTL64(C[MOD5[x + 2]], 1).uint64 ^ C[x].uint64;
			state[x].uint64 = state[x].uint64 ^ D[x].uint64;
			state[x + 5].uint64 = state[x + 5].uint64 ^ D[x].uint64;
			state[x + 10].uint64 = state[x + 10].uint64 ^ D[x].uint64;
			state[x + 15].uint64 = state[x + 15].uint64 ^ D[x].uint64;
			state[x + 20].uint64 = state[x + 20].uint64 ^ D[x].uint64;
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

		for (uint_fast8_t x{ 0u }; x < 25u; x += 5u)
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

		state[0].uint64 = state[0].uint64 ^ Keccak_f1600_RC[i];
	}

	C[1] = xor5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = xor5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = xor5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = xor5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = xor5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL64(C[2], 1);
	D[1] = ROTL64(C[3], 1);
	D[2] = ROTL64(C[4], 1);

	state[0] = xor3(state[0], D[0], C[0]);
	state[6] = xor3(state[6], D[1], C[1]);
	state[12] = xor3(state[12], D[2], C[2]);
	state[6] = ROTR64(state[6], 20);
	state[12] = ROTR64(state[12], 21);

	state[0].uint64 = chi(state[0], state[6], state[12]).uint64 ^ Keccak_f1600_RC[23];

	if (bswap_64(state[0]).uint64 <= d_target[0]) // LTE is allowed because d_target is high 64 bits of uint256 (let CPU do the verification)
	{
		(*solutionCount)++;
		if ((*solutionCount) < MAX_SOLUTION_COUNT_DEVICE) solutions[(*solutionCount) - 1] = nonce.uint64;
	}
}

// --------------------------------------------------------------------
// CUDASolver
// --------------------------------------------------------------------

void CUDASolver::pushTarget(std::unique_ptr<Device>& device)
{
	cudaMemcpyToSymbol(d_target, &device->currentHigh64Target, UINT64_LENGTH, 0, cudaMemcpyHostToDevice);

	device->isNewTarget = false;
}

void CUDASolver::pushMessage(std::unique_ptr<Device>& device)
{
	cudaMemcpyToSymbol(d_midState, device->currentMidState.data(), STATE_LENGTH, 0, cudaMemcpyHostToDevice);

	device->isNewMessage = false;
}

void CUDASolver::checkInputs(std::unique_ptr<Device>& device, char *currentChallenge)
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

void CUDASolver::findSolution(int const deviceID)
{
	auto& device = *std::find_if(m_devices.begin(), m_devices.end(), [&](std::unique_ptr<Device>& device) { return device->deviceID == deviceID; });

	if (!device->initialized) return;

	while (!(device->isNewTarget || device->isNewMessage)) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }

	CudaSafeCall(cudaSetDevice(device->deviceID));

	char *c_currentChallenge = (char *)malloc(s_challenge.size());
	strcpy_s(c_currentChallenge, s_challenge.size() + 1, s_challenge.c_str());

	onMessage(device->deviceID, "Info", "Start mining...");
	onMessage(device->deviceID, "Debug", "Threads: " + std::to_string(device->threads()) + " Grid size: " + std::to_string(device->grid().x) + " Block size:" + std::to_string(device->block().x));

	device->mining = true;
	device->hashCount.store(0ull);
	device->hashStartTime.store(std::chrono::steady_clock::now() - std::chrono::milliseconds(500)); // reduce excessive high hashrate reporting at start
	do
	{
		while (m_pause) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }

		checkInputs(device, c_currentChallenge);

		hashMidstate<<<device->grid(), device->block()>>>(device->d_Solutions, device->d_SolutionCount, getNextWorkPosition(device));

		CudaCheckError();

		cudaError_t response = cudaDeviceSynchronize();
		if (response != cudaSuccess)
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

			for (uint32_t i{ 0u }; i < MAX_SOLUTION_COUNT_DEVICE && i < *device->h_SolutionCount; ++i)
			{
				uint64_t const tempSolution{ device->h_Solutions[i] };

				if (tempSolution != 0u && uniqueSolutions.find(tempSolution) == uniqueSolutions.end())
					uniqueSolutions.emplace(tempSolution);
			}

			std::thread t{ &CUDASolver::submitSolutions, this, uniqueSolutions, std::string{ c_currentChallenge }, device->deviceID };
			t.detach();

			std::memset(device->h_SolutionCount, 0u, UINT32_LENGTH);
		}
	} while (device->mining);

	onMessage(device->deviceID, "Info", "Stop mining...");
	device->hashCount.store(0ull);

	CudaSafeCall(cudaFreeHost(device->h_SolutionCount));
	CudaSafeCall(cudaFreeHost(device->h_Solutions));
	CudaSafeCall(cudaDeviceReset());

	device->initialized = false;
	onMessage(device->deviceID, "Info", "Mining stopped.");
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

		if (NvAPI::foundNvAPI64())
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
