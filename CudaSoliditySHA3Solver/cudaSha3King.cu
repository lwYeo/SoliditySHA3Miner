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

/** libkeccak-tiny
*
* A single-file implementation of SHA-3 and SHAKE.
*
* Modified for CUDA processing by lwYeo
* Date: August, 2018
*
* Implementor: David Leon Gil
* License: CC0, attribution kindly requested. Blame taken too,
* but not liability.
*/

#include "cudaSolver.h"
#include "cudaErrorCheck.cu"

typedef union
{
	uint2		uint2;
	uint64_t	uint64;
	uint8_t		uint8[UINT64_LENGTH];
} nonce_t;

__constant__ uint8_t d_message[MESSAGE_LENGTH];
__constant__ uint8_t d_target[UINT256_LENGTH];

/******** The Keccak-f[1600] permutation ********/

/*** Constants. ***/
__constant__ static const uint8_t rho[24] =
{
	1, 3, 6, 10, 15, 21,
	28, 36, 45, 55, 2, 14,
	27, 41, 56, 8, 25, 43,
	62, 18, 39, 61, 20, 44
};

__constant__ static const uint8_t pi[24] =
{
	10, 7, 11, 17, 18, 3,
	5, 16, 8, 21, 24, 4,
	15, 23, 19, 13, 12, 2,
	20, 14, 22, 9, 6, 1
};

/*** Helper macros to unroll the permutation. ***/
#define delim							0x01
#define rate							SPONGE_LENGTH - (256 / 4)
#define rol(x, s)						(((x) << s) | ((x) >> (64 - s)))
#define REPEAT6(e)						e e e e e e
#define REPEAT24(e)						REPEAT6(e e e e)
#define REPEAT5(e)						e e e e e
#define FOR5(v, s, e)					v = 0; REPEAT5(e; v += s;)

/*** Keccak-f[1600] ***/
__device__ __forceinline__ static void keccakf(void *state)
{
	uint64_t *a{ (uint64_t *)state };
	uint64_t b[5]{ 0 };
	uint64_t t{ 0 };
	uint8_t x, y;

#	pragma unroll
	for (uint32_t i{ 0 }; i < 24u; ++i)
	{
		// Theta
		FOR5(x, 1,
			b[x] = 0;
		FOR5(y, 5,
			b[x] ^= a[x + y]; ))
			FOR5(x, 1,
				FOR5(y, 5,
					a[y + x] ^= b[(x + 4) % 5] ^ rol(b[(x + 1) % 5], 1); ))
			// Rho and pi
			t = a[1];
		x = 0;
		REPEAT24(b[0] = a[pi[x]];
		a[pi[x]] = rol(t, rho[x]);
		t = b[0];
		x++; )
			// Chi
			FOR5(y, 5,
				FOR5(x, 1,
					b[x] = a[y + x];)
				FOR5(x, 1,
					a[y + x] = b[x] ^ ((~b[(x + 1) % 5]) & b[(x + 2) % 5]); ))
			// Iota
			a[0] ^= Keccak_f1600_RC[i];
	}
}

/******** The FIPS202-defined functions. ********/

/*** Some helper macros. ***/

//#define P keccakf
#define _(S)							do { S } while (0)
#define FOR(i, ST, L, S)				_(for (size_t i = 0; i < L; i += ST) { S; })

#define mkapply_ds(NAME, S)																				\
	__device__ __forceinline__ static void NAME(uint8_t* dst, const uint8_t* src, size_t len)			\
		{																								\
			FOR(i, 1, len, S);																			\
		}

mkapply_ds(xorin, dst[i] ^= src[i])					// xorin

#define mkapply_sd(NAME, S)																				\
	__device__ __forceinline__ static void NAME(const uint8_t* src, uint8_t* dst, size_t len)			\
	{																									\
		FOR(i, 1, len, S);																				\
	}

mkapply_sd(setout, dst[i] = src[i])					// setout

// Fold keccakf * F over the full blocks of an input
#define foldP(I, L, F)																					\
	while (L >= rate)																					\
	{																									\
		F(sponge, I, rate);																				\
		keccakf(sponge);																				\
		I += rate;																						\
		L -= rate;																						\
	}

__device__ __forceinline__ static void keccak256(uint8_t *digest, uint8_t const *message)
{
	uint8_t sponge[SPONGE_LENGTH]{ 0 };
	uint32_t messageLength{ MESSAGE_LENGTH };
	uint32_t digestLength{ UINT256_LENGTH };

	// Absorb input.
	foldP(message, messageLength, xorin);

	// Xor in the DS and pad frame.
	sponge[messageLength] ^= delim;
	sponge[rate - 1] ^= 0x80;

	// Xor in the last block.
	xorin(sponge, message, messageLength);

	// Apply keccakf
	keccakf(sponge);

	// Squeeze output.
	foldP(digest, digestLength, setout);

	setout(sponge, digest, digestLength);
}

__device__ __forceinline__ static bool islessThan(uint8_t *left, uint8_t *right)
{
	for (uint32_t i{ 0 }; i < UINT256_LENGTH; ++i)
	{
		if (left[i] < right[i]) return true;
		else if (left[i] > right[i]) return false;
	}
	return false;
}

__global__ void hashMessage(uint64_t *__restrict__ solutions, uint32_t *__restrict__ solutionCount, uint32_t maxSolutionCount, uint64_t startPosition)
{
	uint8_t digest[UINT256_LENGTH];
	uint8_t message[MESSAGE_LENGTH];
	memcpy(message, d_message, MESSAGE_LENGTH);

	nonce_t nonce;
	nonce.uint64 = startPosition + (blockDim.x * blockIdx.x + threadIdx.x);
	memcpy(&message[NONCE_POSITION], &nonce, UINT64_LENGTH);

	keccak256(digest, message);

	if (islessThan(digest, d_target))
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
	void CudaSolver::PushTarget(byte32_t *target, const char *errorMessage)
	{
		CudaCheckError(cudaMemcpyToSymbol(d_target, target, UINT256_LENGTH, 0, cudaMemcpyHostToDevice), errorMessage);
	}

	void CudaSolver::PushMessage(message_ut *message, const char *errorMessage)
	{
		CudaCheckError(cudaMemcpyToSymbol(d_message, message, MESSAGE_LENGTH, 0, cudaMemcpyHostToDevice), errorMessage);
	}

	void CudaSolver::HashMessage(DeviceCUDA *device, const char *errorMessage)
	{
		hashMessage<<<device->Grid, device->Block>>>(device->SolutionsDevice, device->SolutionCountDevice, device->MaxSolutionCount, device->WorkPosition);
		CudaSyncAndCheckError(errorMessage);
	}
}