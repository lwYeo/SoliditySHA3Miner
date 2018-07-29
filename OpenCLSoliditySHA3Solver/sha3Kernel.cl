#define OPENCL_PLATFORM_UNKNOWN	0
#define OPENCL_PLATFORM_NVIDIA	1
#define OPENCL_PLATFORM_AMD		2

#ifndef PLATFORM
#define PLATFORM				OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
#pragma OPENCL EXTENSION		cl_amd_media_ops : enable
#endif
#pragma OPENCL EXTENSION		cl_khr_int64_base_atomics : enable

#ifndef COMPUTE
#define COMPUTE					0
#endif

#define MAX_SOLUTION_COUNT		32u
#define STATE_LENGTH			200u

typedef union
{
	uint2		uint2_s;
	ulong		ulong_s;
} nonce_t;

typedef union
{
	uint2		uint2_s[STATE_LENGTH / sizeof(uint2)];
	ulong		ulong_s[STATE_LENGTH / sizeof(ulong)];
	nonce_t	nonce_s[STATE_LENGTH / sizeof(nonce_t)];
} state_t;

__constant static uint2 const Keccak_f1600_RC[24] = {
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

static inline nonce_t bswap64(const nonce_t input)
{
	nonce_t output;

#if PLATFORM == OPENCL_PLATFORM_NVIDIA

	asm("{"
		"  prmt.b32 %0, %3, 0, 0x0123;"
		"  prmt.b32 %1, %2, 0, 0x0123;"
		"}" : "=r"(output.uint2_s.x), "=r"(output.uint2_s.y) : "r"(input.uint2_s.x), "r"(input.uint2_s.y));

#else

	output.ulong_s = as_ulong(as_uchar8(input.ulong_s).s76543210);

#endif

	return output;
}

static inline uint2 ROL2(const uint2 a, const uint offset)
{
#if PLATFORM == OPENCL_PLATFORM_AMD

	if (offset < 33u)
		return amd_bitalign(a.xy, a.yx, 32u - offset);
	else
		return amd_bitalign(a.yx, a.xy, 64u - offset);

#elif PLATFORM == OPENCL_PLATFORM_NVIDIA && COMPUTE >= 35

	uint2 result;
	if (offset < 33u)
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;

#else

	uint2 result;
	if (offset < 33u)
	{
		result.y = ((a.y << (offset)) | (a.x >> (32u - offset)));
		result.x = ((a.x << (offset)) | (a.y >> (32u - offset)));
	}
	else
	{
		result.y = ((a.x << (offset - 32u)) | (a.y >> (64u - offset)));
		result.x = ((a.y << (offset - 32u)) | (a.x >> (64u - offset)));
	}
	return result;

#endif
}

static inline uint2 chi(uint2 const a, uint2 const b, uint2 const c)
{
#if PLATFORM == OPENCL_PLATFORM_NVIDIA && COMPUTE >= 50

	uint2 output;
	asm("{"
		"  lop3.b32 %0, %2, %4, %6, 0xD2;"
		"  lop3.b32 %1, %3, %5, %7, 0xD2;"
		"}" : "=r"(output.x), "=r"(output.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y));
	return output;

#else

	return bitselect(a ^ c, a, b);

#endif
}

static void keccak_first_round(uint2* state, __constant uint2 const* midstate, uint2 const nounce)
{
	uint2 C[5];

	state[2] = midstate[2] ^ ROL2(nounce, 44);
	state[4] = midstate[4] ^ ROL2(nounce, 14);

	state[6] = midstate[6] ^ ROL2(nounce, 20);
	state[9] = midstate[9] ^ ROL2(nounce, 62);

	state[11] = midstate[11] ^ ROL2(nounce, 7);
	state[13] = midstate[13] ^ ROL2(nounce, 8);

	state[15] = midstate[15] ^ ROL2(nounce, 27);
	state[18] = midstate[18] ^ ROL2(nounce, 16);

	state[20] = midstate[20] ^ ROL2(nounce, 63);
	state[21] = midstate[21] ^ ROL2(nounce, 55);
	state[22] = midstate[22] ^ ROL2(nounce, 39);

	state[0] = chi(midstate[0], midstate[1], state[2]);
	state[0] ^= Keccak_f1600_RC[0];
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
}

static void keccak_skip_first_round(uint2* state)
{
	uint2 C[5], D[5];

	for (uint i = 1u; i < 24u; ++i)
	{
		C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
		C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
		C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
		C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
		C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

		D[0] = ROL2(C[1], 1) ^ C[4];
		state[0] ^= D[0];
		state[5] ^= D[0];
		state[10] ^= D[0];
		state[15] ^= D[0];
		state[20] ^= D[0];

		D[0] = ROL2(C[2], 1) ^ C[0];
		state[1] ^= D[0];
		state[6] ^= D[0];
		state[11] ^= D[0];
		state[16] ^= D[0];
		state[21] ^= D[0];

		D[0] = ROL2(C[3], 1) ^ C[1];
		state[2] ^= D[0];
		state[7] ^= D[0];
		state[12] ^= D[0];
		state[17] ^= D[0];
		state[22] ^= D[0];

		D[0] = ROL2(C[4], 1) ^ C[2];
		state[3] ^= D[0];
		state[8] ^= D[0];
		state[13] ^= D[0];
		state[18] ^= D[0];
		state[23] ^= D[0];

		D[0] = ROL2(C[0], 1) ^ C[3];
		state[4] ^= D[0];
		state[9] ^= D[0];
		state[14] ^= D[0];
		state[19] ^= D[0];
		state[24] ^= D[0];

		C[0] = state[1];
		state[1] = ROL2(state[6], 44);
		state[6] = ROL2(state[9], 20);
		state[9] = ROL2(state[22], 61);
		state[22] = ROL2(state[14], 39);
		state[14] = ROL2(state[20], 18);
		state[20] = ROL2(state[2], 62);
		state[2] = ROL2(state[12], 43);
		state[12] = ROL2(state[13], 25);
		state[13] = ROL2(state[19], 8);
		state[19] = ROL2(state[23], 56);
		state[23] = ROL2(state[15], 41);
		state[15] = ROL2(state[4], 27);
		state[4] = ROL2(state[24], 14);
		state[24] = ROL2(state[21], 2);
		state[21] = ROL2(state[8], 55);
		state[8] = ROL2(state[16], 45);
		state[16] = ROL2(state[5], 36);
		state[5] = ROL2(state[3], 28);
		state[3] = ROL2(state[18], 21);
		state[18] = ROL2(state[17], 15);
		state[17] = ROL2(state[11], 10);
		state[11] = ROL2(state[7], 6);
		state[7] = ROL2(state[10], 3);
		state[10] = ROL2(C[0], 1);

		C[0] = state[0];
		C[1] = state[1];
		state[0] = chi(state[0], state[1], state[2]);
		state[0] ^= Keccak_f1600_RC[i];
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
}

__kernel void hashMidstate(
	__constant uint2 const *midstate, __constant ulong const *target, ulong const startPosition,
	__global volatile ulong *restrict solutions, __global volatile uint *solutionCount)
{
	state_t state;
	nonce_t nonce;
	nonce.ulong_s = get_global_id(0) + startPosition;

	keccak_first_round(state.uint2_s, midstate, nonce.uint2_s);

	keccak_skip_first_round(state.uint2_s);

	if (bswap64(state.nonce_s[0]).ulong_s <= target[0]) // LTE is allowed because d_target is high 64 bits of uint256 (let CPU do the verification)
	{
#ifdef cl_khr_int64_base_atomics
		uint position = atomic_inc(&solutionCount[0]);
#else
		uint position = solutionCount[0];
		++solutionCount[0];
#endif
		if (position < (MAX_SOLUTION_COUNT)) solutions[position] = nonce.ulong_s;
	}
}
