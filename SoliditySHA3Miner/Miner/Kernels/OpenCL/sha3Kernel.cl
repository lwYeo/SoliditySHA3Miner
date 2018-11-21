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

#define OPENCL_PLATFORM_UNKNOWN	0
#define OPENCL_PLATFORM_AMD		2

#ifndef PLATFORM
#	define PLATFORM				OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
#	pragma OPENCL EXTENSION		cl_amd_media_ops : enable
#endif

#define STATE_LENGTH			200u

typedef union _nonce_t
{
	uint2		uint2_s;
	ulong		ulong_s;
} nonce_t;

typedef union _state_t
{
	uint2		uint2_s[STATE_LENGTH / sizeof(uint2)];
	ulong		ulong_s[STATE_LENGTH / sizeof(ulong)];
	nonce_t		nonce_s[STATE_LENGTH / sizeof(nonce_t)];
} state_t;

__constant static uint2 const Keccak_f1600_RC[24] =
{
	(uint2)(0x00000001, 0x00000000),
	(uint2)(0x00008082, 0x00000000),
	(uint2)(0x0000808a, 0x80000000),
	(uint2)(0x80008000, 0x80000000),
	(uint2)(0x0000808b, 0x00000000),
	(uint2)(0x80000001, 0x00000000),
	(uint2)(0x80008081, 0x80000000),
	(uint2)(0x00008009, 0x80000000),
	(uint2)(0x0000008a, 0x00000000),
	(uint2)(0x00000088, 0x00000000),
	(uint2)(0x80008009, 0x00000000),
	(uint2)(0x8000000a, 0x00000000),
	(uint2)(0x8000808b, 0x00000000),
	(uint2)(0x0000008b, 0x80000000),
	(uint2)(0x00008089, 0x80000000),
	(uint2)(0x00008003, 0x80000000),
	(uint2)(0x00008002, 0x80000000),
	(uint2)(0x00000080, 0x80000000),
	(uint2)(0x0000800a, 0x00000000),
	(uint2)(0x8000000a, 0x80000000),
	(uint2)(0x80008081, 0x80000000),
	(uint2)(0x00008080, 0x80000000),
	(uint2)(0x80000001, 0x00000000),
	(uint2)(0x80008008, 0x80000000),
};

static inline nonce_t bswap64(nonce_t const input)
{
	nonce_t output;
	output.ulong_s = as_ulong(as_uchar8(input.ulong_s).s76543210);
	return output;
}

static inline uint2 rol_lte32(uint2 const a, int const offset)
{
	uint2 result;

#if PLATFORM == OPENCL_PLATFORM_AMD

	result = amd_bitalign(a.xy, a.yx, 32 - offset);

#else

	result.y = ((a.y << offset) | (a.x >> (32 - offset)));
	result.x = ((a.x << offset) | (a.y >> (32 - offset)));

#endif

	return result;
}

static inline uint2 rol_gt32(uint2 const a, int const offset)
{
	uint2 result;

#if PLATFORM == OPENCL_PLATFORM_AMD

	result = amd_bitalign(a.yx, a.xy, 64 - offset);

#else

	result.y = ((a.x << (offset - 32)) | (a.y >> (64 - offset)));
	result.x = ((a.y << (offset - 32)) | (a.x >> (64 - offset)));

#endif

	return result;
}

static inline uint2 chi(uint2 const a, uint2 const b, uint2 const c)
{
	return bitselect(a ^ c, a, b);
}

static void keccak(uint2* state, __constant uint2 const* midstate, uint2 const nonce)
{
	uint2 C[5], D[5];

	state[2] = midstate[2] ^ rol_gt32(nonce, 44);
	state[4] = midstate[4] ^ rol_lte32(nonce, 14);

	state[6] = midstate[6] ^ rol_lte32(nonce, 20);
	state[9] = midstate[9] ^ rol_gt32(nonce, 62);

	state[11] = midstate[11] ^ rol_lte32(nonce, 7);
	state[13] = midstate[13] ^ rol_lte32(nonce, 8);

	state[15] = midstate[15] ^ rol_lte32(nonce, 27);
	state[18] = midstate[18] ^ rol_lte32(nonce, 16);

	state[20] = midstate[20] ^ rol_gt32(nonce, 63);
	state[21] = midstate[21] ^ rol_gt32(nonce, 55);
	state[22] = midstate[22] ^ rol_gt32(nonce, 39);

	state[0] = chi(midstate[0], midstate[1], state[2]) ^ Keccak_f1600_RC[0];
	state[1] = chi(midstate[1], state[2], midstate[3]);
	state[2] = chi(state[2], midstate[3], state[4]);
	state[3] = chi(midstate[3], state[4], midstate[0]);
	state[4] = chi(state[4], midstate[0], midstate[1]);

	C[0] = state[6];
	state[5] = chi(midstate[5], C[0], midstate[7]);
	state[6] = chi(C[0], midstate[7], midstate[8]);
	state[7] = chi(midstate[7], midstate[8], state[9]);
	state[8] = chi(midstate[8], state[9], midstate[5]);
	state[9] = chi(state[9], midstate[5], C[0]);

	C[0] = state[11];
	state[10] = chi(midstate[10], C[0], midstate[12]);
	state[11] = chi(C[0], midstate[12], state[13]);
	state[12] = chi(midstate[12], state[13], midstate[14]);
	state[13] = chi(state[13], midstate[14], midstate[10]);
	state[14] = chi(midstate[14], midstate[10], C[0]);

	C[0] = state[15];
	state[15] = chi(C[0], midstate[16], midstate[17]);
	state[16] = chi(midstate[16], midstate[17], state[18]);
	state[17] = chi(midstate[17], state[18], midstate[19]);
	state[18] = chi(state[18], midstate[19], C[0]);
	state[19] = chi(midstate[19], C[0], midstate[16]);

	C[0] = state[20];
	C[1] = state[21];
	state[20] = chi(C[0], C[1], state[22]);
	state[21] = chi(C[1], state[22], midstate[23]);
	state[22] = chi(state[22], midstate[23], midstate[24]);
	state[23] = chi(midstate[23], midstate[24], C[0]);
	state[24] = chi(midstate[24], C[0], C[1]);

#	pragma unroll
	for (int i = 1; i < 23; ++i)
	{
		C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
		C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
		C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
		C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
		C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

		D[0] = rol_lte32(C[1], 1) ^ C[4];
		state[0] ^= D[0];
		state[5] ^= D[0];
		state[10] ^= D[0];
		state[15] ^= D[0];
		state[20] ^= D[0];

		D[0] = rol_lte32(C[2], 1) ^ C[0];
		state[1] ^= D[0];
		state[6] ^= D[0];
		state[11] ^= D[0];
		state[16] ^= D[0];
		state[21] ^= D[0];

		D[0] = rol_lte32(C[3], 1) ^ C[1];
		state[2] ^= D[0];
		state[7] ^= D[0];
		state[12] ^= D[0];
		state[17] ^= D[0];
		state[22] ^= D[0];

		D[0] = rol_lte32(C[4], 1) ^ C[2];
		state[3] ^= D[0];
		state[8] ^= D[0];
		state[13] ^= D[0];
		state[18] ^= D[0];
		state[23] ^= D[0];

		D[0] = rol_lte32(C[0], 1) ^ C[3];
		state[4] ^= D[0];
		state[9] ^= D[0];
		state[14] ^= D[0];
		state[19] ^= D[0];
		state[24] ^= D[0];

		C[0] = state[1];
		state[1] = rol_gt32(state[6], 44);
		state[6] = rol_lte32(state[9], 20);
		state[9] = rol_gt32(state[22], 61);
		state[22] = rol_gt32(state[14], 39);
		state[14] = rol_lte32(state[20], 18);
		state[20] = rol_gt32(state[2], 62);
		state[2] = rol_gt32(state[12], 43);
		state[12] = rol_lte32(state[13], 25);
		state[13] = rol_lte32(state[19], 8);
		state[19] = rol_gt32(state[23], 56);
		state[23] = rol_gt32(state[15], 41);
		state[15] = rol_lte32(state[4], 27);
		state[4] = rol_lte32(state[24], 14);
		state[24] = rol_lte32(state[21], 2);
		state[21] = rol_gt32(state[8], 55);
		state[8] = rol_gt32(state[16], 45);
		state[16] = rol_gt32(state[5], 36);
		state[5] = rol_lte32(state[3], 28);
		state[3] = rol_lte32(state[18], 21);
		state[18] = rol_lte32(state[17], 15);
		state[17] = rol_lte32(state[11], 10);
		state[11] = rol_lte32(state[7], 6);
		state[7] = rol_lte32(state[10], 3);
		state[10] = rol_lte32(C[0], 1);

		C[0] = state[0];
		C[1] = state[1];
		state[0] = chi(state[0], state[1], state[2]) ^ Keccak_f1600_RC[i];
		state[1] = chi(state[1], state[2], state[3]);
		state[2] = chi(state[2], state[3], state[4]);
		state[3] = chi(state[3], state[4], C[0]);
		state[4] = chi(state[4], C[0], C[1]);

		C[0] = state[5];
		C[1] = state[6];
		state[5] = chi(state[5], state[6], state[7]);
		state[6] = chi(state[6], state[7], state[8]);
		state[7] = chi(state[7], state[8], state[9]);
		state[8] = chi(state[8], state[9], C[0]);
		state[9] = chi(state[9], C[0], C[1]);

		C[0] = state[10];
		C[1] = state[11];
		state[10] = chi(state[10], state[11], state[12]);
		state[11] = chi(state[11], state[12], state[13]);
		state[12] = chi(state[12], state[13], state[14]);
		state[13] = chi(state[13], state[14], C[0]);
		state[14] = chi(state[14], C[0], C[1]);

		C[0] = state[15];
		C[1] = state[16];
		state[15] = chi(state[15], state[16], state[17]);
		state[16] = chi(state[16], state[17], state[18]);
		state[17] = chi(state[17], state[18], state[19]);
		state[18] = chi(state[18], state[19], C[0]);
		state[19] = chi(state[19], C[0], C[1]);

		C[0] = state[20];
		C[1] = state[21];
		state[20] = chi(state[20], state[21], state[22]);
		state[21] = chi(state[21], state[22], state[23]);
		state[22] = chi(state[22], state[23], state[24]);
		state[23] = chi(state[23], state[24], C[0]);
		state[24] = chi(state[24], C[0], C[1]);
	}

	C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
	C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
	C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
	C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
	C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

	D[0] = rol_lte32(C[1], 1) ^ C[4];
	state[0] ^= D[0];
	state[5] ^= D[0];
	state[10] ^= D[0];
	state[15] ^= D[0];
	state[20] ^= D[0];

	D[0] = rol_lte32(C[2], 1) ^ C[0];
	state[1] ^= D[0];
	state[6] ^= D[0];
	state[11] ^= D[0];
	state[16] ^= D[0];
	state[21] ^= D[0];

	D[0] = rol_lte32(C[3], 1) ^ C[1];
	state[2] ^= D[0];
	state[7] ^= D[0];
	state[12] ^= D[0];
	state[17] ^= D[0];
	state[22] ^= D[0];

	D[0] = rol_lte32(C[4], 1) ^ C[2];
	state[3] ^= D[0];
	state[8] ^= D[0];
	state[13] ^= D[0];
	state[18] ^= D[0];
	state[23] ^= D[0];

	D[0] = rol_lte32(C[0], 1) ^ C[3];
	state[4] ^= D[0];
	state[9] ^= D[0];
	state[14] ^= D[0];
	state[19] ^= D[0];
	state[24] ^= D[0];

	C[0] = state[1];
	state[1] = rol_gt32(state[6], 44);
	state[6] = rol_lte32(state[9], 20);
	state[9] = rol_gt32(state[22], 61);
	state[22] = rol_gt32(state[14], 39);
	state[14] = rol_lte32(state[20], 18);
	state[20] = rol_gt32(state[2], 62);
	state[2] = rol_gt32(state[12], 43);
	state[12] = rol_lte32(state[13], 25);
	state[13] = rol_lte32(state[19], 8);
	state[19] = rol_gt32(state[23], 56);
	state[23] = rol_gt32(state[15], 41);
	state[15] = rol_lte32(state[4], 27);
	state[4] = rol_lte32(state[24], 14);
	state[24] = rol_lte32(state[21], 2);
	state[21] = rol_gt32(state[8], 55);
	state[8] = rol_gt32(state[16], 45);
	state[16] = rol_gt32(state[5], 36);
	state[5] = rol_lte32(state[3], 28);
	state[3] = rol_lte32(state[18], 21);
	state[18] = rol_lte32(state[17], 15);
	state[17] = rol_lte32(state[11], 10);
	state[11] = rol_lte32(state[7], 6);
	state[7] = rol_lte32(state[10], 3);
	state[10] = rol_lte32(C[0], 1);

	state[0] = chi(state[0], state[1], state[2]) ^ Keccak_f1600_RC[23];
}

__kernel void hashMidstate(
	__constant uint2 const *midstate, __constant ulong const *target,
	ulong const startPosition, uint const maxSolutionCount,
	__global volatile ulong *restrict solutions, __global volatile uint *solutionCount)
{
	state_t state;
	nonce_t nonce;
	nonce.ulong_s = startPosition + get_global_id(0);

	keccak(state.uint2_s, midstate, nonce.uint2_s);

	if (bswap64(state.nonce_s[0]).ulong_s <= target[0]) // LTE is allowed because target is high 64 bits of uint256 (let CPU do the verification)
	{
		if (solutionCount[0] < maxSolutionCount)
		{
			solutions[solutionCount[0]] = nonce.ulong_s;
			++solutionCount[0];
		}
	}
}