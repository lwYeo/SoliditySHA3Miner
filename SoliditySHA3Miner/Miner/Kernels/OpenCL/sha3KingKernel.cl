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
* Modified for openCL processing by lwYeo
* Date: August, 2018
*
* Implementor: David Leon Gil
* License: CC0, attribution kindly requested. Blame taken too,
* but not liability.
*/

/******** The Keccak-f[1600] permutation ********/

#define OPENCL_PLATFORM_UNKNOWN	0
#define OPENCL_PLATFORM_AMD		2

#ifndef PLATFORM
#	define PLATFORM				OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
#	pragma OPENCL EXTENSION		cl_amd_media_ops : enable
#endif

#define ADDRESS_LENGTH			20u
#define UINT64_LENGTH			8u
#define UINT256_LENGTH			32u
#define MESSAGE_LENGTH			84u
#define SPONGE_LENGTH			200u
#define NONCE_POSITION			UINT256_LENGTH + ADDRESS_LENGTH + ADDRESS_LENGTH

typedef union _nonce_t
{
	uint2		uint2_t;
	ulong		uint64_t;
	uchar		uint8_t[UINT64_LENGTH];
} nonce_t;

static inline ulong rol(const ulong x, const uint s)
{
#if PLATFORM == OPENCL_PLATFORM_AMD

	uint2 output;
	uint2 x2 = as_uint2(x);

	output = (s > 32u) ? amd_bitalign((x2).yx, (x2).xy, 64u - s) : amd_bitalign((x2).xy, (x2).yx, 32u - s);
	return as_ulong(output);

#else

	return (((x) << s) | ((x) >> (64u - s)));

#endif
}

#define delim										0x01
#define rate										SPONGE_LENGTH - (256 / 4)

/*** Constants. ***/
__constant static ulong const Keccak_f1600_RC[24] =
{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__constant static const uchar rho[24] =
{
	1, 3, 6, 10, 15, 21,
	28, 36, 45, 55, 2, 14,
	27, 41, 56, 8, 25, 43,
	62, 18, 39, 61, 20, 44
};

__constant static const uchar pi[24] =
{
	10, 7, 11, 17, 18, 3,
	5, 16, 8, 21, 24, 4,
	15, 23, 19, 13, 12, 2,
	20, 14, 22, 9, 6, 1
};

static inline void setout(const uchar* src, uchar* dst, size_t len)
{
	for (size_t i = 0; i < len; i += 1)
		dst[i] = src[i];
}

static inline void xorin(uchar* dst, const uchar* src, size_t len)
{
	for (size_t i = 0; i < len; i += 1)
		dst[i] ^= src[i];
}

/*** This is the unrolled version of the original macro ***/
static inline void keccakf(void *state)
{
	ulong *a = (ulong *)state;
	ulong b[5] = { 0, 0, 0, 0, 0 };
	ulong t;

#	pragma unroll
	for (uint i = 0; i < 24u; ++i)
	{
		// Theta
		b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
		b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
		b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
		b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
		b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

		a[0] ^= b[4] ^ rol(b[1], 1);
		a[5] ^= b[4] ^ rol(b[1], 1);
		a[10] ^= b[4] ^ rol(b[1], 1);
		a[15] ^= b[4] ^ rol(b[1], 1);
		a[20] ^= b[4] ^ rol(b[1], 1);

		a[1] ^= b[0] ^ rol(b[2], 1);
		a[6] ^= b[0] ^ rol(b[2], 1);
		a[11] ^= b[0] ^ rol(b[2], 1);
		a[16] ^= b[0] ^ rol(b[2], 1);
		a[21] ^= b[0] ^ rol(b[2], 1);

		a[2] ^= b[1] ^ rol(b[3], 1);
		a[7] ^= b[1] ^ rol(b[3], 1);
		a[12] ^= b[1] ^ rol(b[3], 1);
		a[17] ^= b[1] ^ rol(b[3], 1);
		a[22] ^= b[1] ^ rol(b[3], 1);

		a[3] ^= b[2] ^ rol(b[4], 1);
		a[8] ^= b[2] ^ rol(b[4], 1);
		a[13] ^= b[2] ^ rol(b[4], 1);
		a[18] ^= b[2] ^ rol(b[4], 1);
		a[23] ^= b[2] ^ rol(b[4], 1);

		a[4] ^= b[3] ^ rol(b[0], 1);
		a[9] ^= b[3] ^ rol(b[0], 1);
		a[14] ^= b[3] ^ rol(b[0], 1);
		a[19] ^= b[3] ^ rol(b[0], 1);
		a[24] ^= b[3] ^ rol(b[0], 1);

		// Rho Pi
		t = a[1];
		b[0] = a[pi[0]];
		a[pi[0]] = rol(t, rho[0]);

		t = b[0];
		b[0] = a[pi[1]];
		a[pi[1]] = rol(t, rho[1]);

		t = b[0];
		b[0] = a[pi[2]];
		a[pi[2]] = rol(t, rho[2]);

		t = b[0];
		b[0] = a[pi[3]];
		a[pi[3]] = rol(t, rho[3]);

		t = b[0];
		b[0] = a[pi[4]];
		a[pi[4]] = rol(t, rho[4]);

		t = b[0];
		b[0] = a[pi[5]];
		a[pi[5]] = rol(t, rho[5]);

		t = b[0];
		b[0] = a[pi[6]];
		a[pi[6]] = rol(t, rho[6]);

		t = b[0];
		b[0] = a[pi[7]];
		a[pi[7]] = rol(t, rho[7]);

		t = b[0];
		b[0] = a[pi[8]];
		a[pi[8]] = rol(t, rho[8]);

		t = b[0];
		b[0] = a[pi[9]];
		a[pi[9]] = rol(t, rho[9]);

		t = b[0];
		b[0] = a[pi[10]];
		a[pi[10]] = rol(t, rho[10]);

		t = b[0];
		b[0] = a[pi[11]];
		a[pi[11]] = rol(t, rho[11]);

		t = b[0];
		b[0] = a[pi[12]];
		a[pi[12]] = rol(t, rho[12]);

		t = b[0];
		b[0] = a[pi[13]];
		a[pi[13]] = rol(t, rho[13]);

		t = b[0];
		b[0] = a[pi[14]];
		a[pi[14]] = rol(t, rho[14]);

		t = b[0];
		b[0] = a[pi[15]];
		a[pi[15]] = rol(t, rho[15]);

		t = b[0];
		b[0] = a[pi[16]];
		a[pi[16]] = rol(t, rho[16]);

		t = b[0];
		b[0] = a[pi[17]];
		a[pi[17]] = rol(t, rho[17]);

		t = b[0];
		b[0] = a[pi[18]];
		a[pi[18]] = rol(t, rho[18]);

		t = b[0];
		b[0] = a[pi[19]];
		a[pi[19]] = rol(t, rho[19]);

		t = b[0];
		b[0] = a[pi[20]];
		a[pi[20]] = rol(t, rho[20]);

		t = b[0];
		b[0] = a[pi[21]];
		a[pi[21]] = rol(t, rho[21]);

		t = b[0];
		b[0] = a[pi[22]];
		a[pi[22]] = rol(t, rho[22]);

		t = b[0];
		b[0] = a[pi[23]];
		a[pi[23]] = rol(t, rho[23]);

		// Chi
		b[0] = a[0];
		b[1] = a[1];
		b[2] = a[2];
		b[3] = a[3];
		b[4] = a[4];
		a[0] = b[0] ^ ((~b[1]) & b[2]);
		a[1] = b[1] ^ ((~b[2]) & b[3]);
		a[2] = b[2] ^ ((~b[3]) & b[4]);
		a[3] = b[3] ^ ((~b[4]) & b[0]);
		a[4] = b[4] ^ ((~b[0]) & b[1]);

		b[0] = a[5];
		b[1] = a[6];
		b[2] = a[7];
		b[3] = a[8];
		b[4] = a[9];
		a[5] = b[0] ^ ((~b[1]) & b[2]);
		a[6] = b[1] ^ ((~b[2]) & b[3]);
		a[7] = b[2] ^ ((~b[3]) & b[4]);
		a[8] = b[3] ^ ((~b[4]) & b[0]);
		a[9] = b[4] ^ ((~b[0]) & b[1]);

		b[0] = a[10];
		b[1] = a[11];
		b[2] = a[12];
		b[3] = a[13];
		b[4] = a[14];
		a[10] = b[0] ^ ((~b[1]) & b[2]);
		a[11] = b[1] ^ ((~b[2]) & b[3]);
		a[12] = b[2] ^ ((~b[3]) & b[4]);
		a[13] = b[3] ^ ((~b[4]) & b[0]);
		a[14] = b[4] ^ ((~b[0]) & b[1]);

		b[0] = a[15];
		b[1] = a[16];
		b[2] = a[17];
		b[3] = a[18];
		b[4] = a[19];
		a[15] = b[0] ^ ((~b[1]) & b[2]);
		a[16] = b[1] ^ ((~b[2]) & b[3]);
		a[17] = b[2] ^ ((~b[3]) & b[4]);
		a[18] = b[3] ^ ((~b[4]) & b[0]);
		a[19] = b[4] ^ ((~b[0]) & b[1]);

		b[0] = a[20];
		b[1] = a[21];
		b[2] = a[22];
		b[3] = a[23];
		b[4] = a[24];
		a[20] = b[0] ^ ((~b[1]) & b[2]);
		a[21] = b[1] ^ ((~b[2]) & b[3]);
		a[22] = b[2] ^ ((~b[3]) & b[4]);
		a[23] = b[3] ^ ((~b[4]) & b[0]);
		a[24] = b[4] ^ ((~b[0]) & b[1]);

		// Iota
		a[0] ^= Keccak_f1600_RC[i];
	}
}

static inline void keccak256(uchar *digest, uchar const *message)
{
	uchar sponge[SPONGE_LENGTH];
	uint messageLength = MESSAGE_LENGTH;
	uint digestLength = UINT256_LENGTH;

	for (uchar i = 0; i < SPONGE_LENGTH; ++i)
		sponge[i] = 0;

	// Absorb input
	while (messageLength >= (SPONGE_LENGTH - 64u))
	{
		xorin(sponge, message, SPONGE_LENGTH - 64u);
		keccakf(sponge);
		message += (SPONGE_LENGTH - 64u);
		messageLength -= (SPONGE_LENGTH - 64u);
	}

	// Xor in the DS and pad frame
	sponge[messageLength] ^= 0x01u;
	sponge[SPONGE_LENGTH - 65u] ^= 0x80u;

	// Xor in the last block
	xorin(sponge, message, messageLength);

	// Apply keccakf
	keccakf(sponge);

	// Squeeze output
	while (digestLength >= (SPONGE_LENGTH - 64u))
	{
		setout(sponge, digest, SPONGE_LENGTH - 64u);
		keccakf(sponge);
		digest += (SPONGE_LENGTH - 64u);
		digestLength -= (SPONGE_LENGTH - 64u);
	};

	setout(sponge, digest, digestLength);
}

static inline bool islessThan(uchar const *left, __constant uchar const *right)
{
	for (uchar i = 0; i < UINT256_LENGTH; ++i)
	{
		if (left[i] < right[i]) return true;
		else if (left[i] > right[i]) return false;
	}
	return false;
}

__kernel void hashMessage(
	__constant uchar const *d_message, __constant uchar const *d_target,
	ulong const startPosition, uint const maxSolutionCount,
	__global volatile ulong *restrict solutions, __global volatile uint *solutionCount)
{
	uchar digest[UINT256_LENGTH];

	uchar message[MESSAGE_LENGTH];
	for (uchar i = 0; i < MESSAGE_LENGTH; ++i)
		message[i] = d_message[i];

	nonce_t nonce;
	nonce.uint64_t = startPosition + get_global_id(0);

	for (uchar i = 0; i < UINT64_LENGTH; ++i)
		message[NONCE_POSITION + i] = nonce.uint8_t[i];

	keccak256(digest, message);

	if (islessThan(digest, d_target))
	{
		if (solutionCount[0] < (maxSolutionCount))
		{
			solutions[solutionCount[0]] = nonce.uint64_t;
			++solutionCount[0];
		}
	}
}