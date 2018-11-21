/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
* based off of https://github.com/Dunhili/SHA3-gpu-brute-force-cracker/blob/master/sha3.cu
*
* Author: Brian Bowden
* Date: 5/12/14
*
* This is the parallel version of SHA-3.
*/

#include "cudaErrorCheck.cu"
#include "cudaSolver.h"

typedef union
{
	uint2		uint2;
	uint64_t	uint64;
	uint8_t		uint8[UINT64_LENGTH];
} nonce_t;

__constant__ uint64_t d_midstate[25];
__constant__ uint64_t d_target[1];

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
#if __CUDA_ARCH__ >= 500
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0x96;"
		"  lop3.b32 %1, %3, %5, %7, 0x96;"
		"  lop3.b32 %0, %0, %8, %10, 0x96;"
		"  lop3.b32 %1, %1, %9, %11, 0x96;"
		"}" : "=r"(output.uint2.x), "=r"(output.uint2.y)
		: "r"(a.uint2.x), "r"(a.uint2.y), "r"(b.uint2.x), "r"(b.uint2.y), "r"(c.uint2.x), "r"(c.uint2.y), "r"(d.uint2.x), "r"(d.uint2.y), "r"(e.uint2.x), "r"(e.uint2.y));
#else
	asm("{"
		"  xor.b64 %0, %1, %2;"
		"  xor.b64 %0, %0, %3;"
		"  xor.b64 %0, %0, %4;"
		"  xor.b64 %0, %0, %5;"
		"}" : "=l"(output.uint64) : "l"(a.uint64), "l"(b.uint64), "l"(c.uint64), "l"(d.uint64), "l"(e.uint64));
#endif
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

__device__ __forceinline__ nonce_t rotl(nonce_t input, uint32_t const offset)
{
#if __CUDA_ARCH__ >= 320
	asm("{"
		"  .reg .b32 tmp;"
		"  shf.l.wrap.b32 tmp, %1, %0, %2;"
		"  shf.l.wrap.b32 %1, %0, %1, %2;"
		"  mov.b32 %0, tmp;"
		"}" : "+r"(input.uint2.x), "+r"(input.uint2.y) : "r"(offset));
#else
	input.uint64 = (input.uint64 << offset) ^ (input.uint64 >> (64u - offset));
#endif
	return input;
}

__device__ __forceinline__ nonce_t rotr(nonce_t input, uint32_t const offset)
{
#if __CUDA_ARCH__ >= 320
	asm("{"
		"  .reg .b32 tmp;"
		"  shf.r.wrap.b32 tmp, %0, %1, %2;"
		"  shf.r.wrap.b32 %1, %1, %0, %2;"
		"  mov.b32 %0, tmp;"
		"}" : "+r"(input.uint2.x), "+r"(input.uint2.y) : "r"(offset));
#else
	input.uint64 = (input.uint64 >> offset) ^ (input.uint64 << (64u - offset));
#endif
	return input;
}

__global__ void hashMidstate(uint64_t *__restrict__ solutions, uint32_t *__restrict__ solutionCount, uint32_t maxSolutionCount, uint64_t startPosition)
{
	nonce_t nonce, state[25], C[5], D[5], n[11];
	nonce.uint64 = blockDim.x * blockIdx.x + threadIdx.x + startPosition;

	n[0] = rotl(nonce, 7);
	n[1] = rotl(n[0], 1);
	n[2] = rotl(n[1], 6);
	n[3] = rotl(n[2], 2);
	n[4] = rotl(n[3], 4);
	n[5] = rotl(n[4], 7);
	n[6] = rotl(n[5], 12);
	n[7] = rotl(n[6], 5);
	n[8] = rotl(n[7], 11);
	n[9] = rotl(n[8], 7);
	n[10] = rotl(n[9], 1);

	C[0].uint64 = d_midstate[0];
	C[1].uint64 = d_midstate[1];
	C[2].uint64 = d_midstate[2] ^ n[7].uint64;
	C[3].uint64 = d_midstate[3];
	C[4].uint64 = d_midstate[4] ^ n[2].uint64;
	state[0].uint64 = chi(C[0], C[1], C[2]).uint64 ^ Keccak_f1600_RC[0];
	state[1] = chi(C[1], C[2], C[3]);
	state[2] = chi(C[2], C[3], C[4]);
	state[3] = chi(C[3], C[4], C[0]);
	state[4] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midstate[5];
	C[1].uint64 = d_midstate[6] ^ n[4].uint64;
	C[2].uint64 = d_midstate[7];
	C[3].uint64 = d_midstate[8];
	C[4].uint64 = d_midstate[9] ^ n[9].uint64;
	state[5] = chi(C[0], C[1], C[2]);
	state[6] = chi(C[1], C[2], C[3]);
	state[7] = chi(C[2], C[3], C[4]);
	state[8] = chi(C[3], C[4], C[0]);
	state[9] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midstate[10];
	C[1].uint64 = d_midstate[11] ^ n[0].uint64;
	C[2].uint64 = d_midstate[12];
	C[3].uint64 = d_midstate[13] ^ n[1].uint64;
	C[4].uint64 = d_midstate[14];
	state[10] = chi(C[0], C[1], C[2]);
	state[11] = chi(C[1], C[2], C[3]);
	state[12] = chi(C[2], C[3], C[4]);
	state[13] = chi(C[3], C[4], C[0]);
	state[14] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midstate[15] ^ n[5].uint64;
	C[1].uint64 = d_midstate[16];
	C[2].uint64 = d_midstate[17];
	C[3].uint64 = d_midstate[18] ^ n[3].uint64;
	C[4].uint64 = d_midstate[19];
	state[15] = chi(C[0], C[1], C[2]);
	state[16] = chi(C[1], C[2], C[3]);
	state[17] = chi(C[2], C[3], C[4]);
	state[18] = chi(C[3], C[4], C[0]);
	state[19] = chi(C[4], C[0], C[1]);

	C[0].uint64 = d_midstate[20] ^ n[10].uint64;
	C[1].uint64 = d_midstate[21] ^ n[8].uint64;
	C[2].uint64 = d_midstate[22] ^ n[6].uint64;
	C[3].uint64 = d_midstate[23];
	C[4].uint64 = d_midstate[24];
	state[20] = chi(C[0], C[1], C[2]);
	state[21] = chi(C[1], C[2], C[3]);
	state[22] = chi(C[2], C[3], C[4]);
	state[23] = chi(C[3], C[4], C[0]);
	state[24] = chi(C[4], C[0], C[1]);

#if __CUDA_ARCH__ >= 350
#	pragma unroll
#endif
	for (int i{ 1 }; i < 23; ++i)
	{
		C[1] = xor5(state[0], state[5], state[10], state[15], state[20]);
		C[2] = xor5(state[1], state[6], state[11], state[16], state[21]);
		C[3] = xor5(state[2], state[7], state[12], state[17], state[22]);
		C[4] = xor5(state[3], state[8], state[13], state[18], state[23]);
		C[0] = xor5(state[4], state[9], state[14], state[19], state[24]);

#if __CUDA_ARCH__ >= 350
		D[0] = rotl(C[2], 1);
		state[0] = xor3(state[0], D[0], C[0]);
		state[5] = xor3(state[5], D[0], C[0]);
		state[10] = xor3(state[10], D[0], C[0]);
		state[15] = xor3(state[15], D[0], C[0]);
		state[20] = xor3(state[20], D[0], C[0]);

		D[1] = rotl(C[3], 1);
		state[1] = xor3(state[1], D[1], C[1]);
		state[6] = xor3(state[6], D[1], C[1]);
		state[11] = xor3(state[11], D[1], C[1]);
		state[16] = xor3(state[16], D[1], C[1]);
		state[21] = xor3(state[21], D[1], C[1]);

		D[2] = rotl(C[4], 1);
		state[2] = xor3(state[2], D[2], C[2]);
		state[7] = xor3(state[7], D[2], C[2]);
		state[12] = xor3(state[12], D[2], C[2]);
		state[17] = xor3(state[17], D[2], C[2]);
		state[22] = xor3(state[22], D[2], C[2]);

		D[3] = rotl(C[0], 1);
		state[3] = xor3(state[3], D[3], C[3]);
		state[8] = xor3(state[8], D[3], C[3]);
		state[13] = xor3(state[13], D[3], C[3]);
		state[18] = xor3(state[18], D[3], C[3]);
		state[23] = xor3(state[23], D[3], C[3]);

		D[4] = rotl(C[1], 1);
		state[4] = xor3(state[4], D[4], C[4]);
		state[9] = xor3(state[9], D[4], C[4]);
		state[14] = xor3(state[14], D[4], C[4]);
		state[19] = xor3(state[19], D[4], C[4]);
		state[24] = xor3(state[24], D[4], C[4]);
#else
		for (int x{ 0 }; x < 5; ++x)
		{
			D[x].uint64 = rotl(C[(x + 2) % 5], 1).uint64 ^ C[x].uint64;
			state[x].uint64 = state[x].uint64 ^ D[x].uint64;
			state[x + 5].uint64 = state[x + 5].uint64 ^ D[x].uint64;
			state[x + 10].uint64 = state[x + 10].uint64 ^ D[x].uint64;
			state[x + 15].uint64 = state[x + 15].uint64 ^ D[x].uint64;
			state[x + 20].uint64 = state[x + 20].uint64 ^ D[x].uint64;
		}
#endif

		C[0] = state[1];
		state[1] = rotr(state[6], 20);
		state[6] = rotl(state[9], 20);
		state[9] = rotr(state[22], 3);
		state[22] = rotr(state[14], 25);
		state[14] = rotl(state[20], 18);
		state[20] = rotr(state[2], 2);
		state[2] = rotr(state[12], 21);
		state[12] = rotl(state[13], 25);
		state[13] = rotl(state[19], 8);
		state[19] = rotr(state[23], 8);
		state[23] = rotr(state[15], 23);
		state[15] = rotl(state[4], 27);
		state[4] = rotl(state[24], 14);
		state[24] = rotl(state[21], 2);
		state[21] = rotr(state[8], 9);
		state[8] = rotr(state[16], 19);
		state[16] = rotr(state[5], 28);
		state[5] = rotl(state[3], 28);
		state[3] = rotl(state[18], 21);
		state[18] = rotl(state[17], 15);
		state[17] = rotl(state[11], 10);
		state[11] = rotl(state[7], 6);
		state[7] = rotl(state[10], 3);
		state[10] = rotl(C[0], 1);

#if __CUDA_ARCH__ >= 350
#	pragma unroll
#endif
		for (int x{ 0 }; x < 25; x += 5)
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

	D[0] = rotl(C[2], 1);
	D[1] = rotl(C[3], 1);
	D[2] = rotl(C[4], 1);

	state[0] = xor3(state[0], D[0], C[0]);
	state[6] = xor3(state[6], D[1], C[1]);
	state[12] = xor3(state[12], D[2], C[2]);
	state[6] = rotr(state[6], 20);
	state[12] = rotr(state[12], 21);

	state[0].uint64 = chi(state[0], state[6], state[12]).uint64 ^ Keccak_f1600_RC[23];

	if (bswap_64(state[0]).uint64 <= d_target[0]) // LTE is allowed because d_target is high 64 bits of uint256 (let CPU do the verification)
	{
		if (*solutionCount < maxSolutionCount)
		{
			solutions[*solutionCount] = nonce.uint64;
			(*solutionCount)++;
		}
	}
}

// --------------------------------------------------------------------
// CudaSolver
// --------------------------------------------------------------------

namespace CUDASolver
{
	void CudaSolver::PushHigh64Target(uint64_t *high64Target, const char *errorMessage)
	{
		CudaCheckError(cudaMemcpyToSymbol(d_target, high64Target, UINT64_LENGTH, 0, cudaMemcpyHostToDevice), errorMessage);
	}

	void CudaSolver::PushMidState(sponge_ut *midState, const char *errorMessage)
	{
		CudaCheckError(cudaMemcpyToSymbol(d_midstate, midState, SPONGE_LENGTH, 0, cudaMemcpyHostToDevice), errorMessage);
	}

	void CudaSolver::HashMidState(DeviceCUDA *device, const char *errorMessage)
	{
		hashMidstate<<<device->Grid, device->Block>>>(device->SolutionsDevice, device->SolutionCountDevice, device->MaxSolutionCount, device->WorkPosition);
		CudaSyncAndCheckError(errorMessage);
	}
}