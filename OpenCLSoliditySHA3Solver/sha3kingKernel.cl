#define OPENCL_PLATFORM_UNKNOWN	0
#define OPENCL_PLATFORM_NVIDIA	1
#define OPENCL_PLATFORM_AMD		2

#ifndef PLATFORM
#define PLATFORM				OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
#pragma OPENCL EXTENSION		cl_amd_media_ops : enable
#endif

#ifndef COMPUTE
#define COMPUTE					0
#endif

#define MAX_SOLUTION_COUNT		32u
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

/*** Helper macros to unroll the permutation. ***/
#define delim							0x01
#define rate							SPONGE_LENGTH - (256 / 4)
#define rol(x, s)						(((x) << s) | ((x) >> (64 - s)))
#define REPEAT6(e)						e e e e e e
#define REPEAT24(e)						REPEAT6(e e e e)
#define REPEAT5(e)						e e e e e
#define FOR5(v, s, e)					v = 0; REPEAT5(e; v += s;)

/*** Keccak-f[1600] ***/
static inline void keccakf(void *state)
{
	ulong *a = (ulong *)state;
	ulong b[5] = { 0, 0, 0, 0, 0 };
	ulong t = 0;
	uchar x, y;

#	pragma unroll 8
	for (uchar i = 0; i < 24u; ++i)
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
	static inline void NAME(uchar* dst, const uchar* src, size_t len)									\
		{																								\
			FOR(i, 1, len, S);																			\
		}

mkapply_ds(xorin, dst[i] ^= src[i])					// xorin

#define mkapply_sd(NAME, S)																				\
	static inline void NAME(const uchar* src, uchar* dst, size_t len)									\
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

static inline void keccak256(uchar *digest, uchar const *message)
{
	ulong messageLength = MESSAGE_LENGTH;
	ulong digestLength = UINT256_LENGTH;
	uchar sponge[SPONGE_LENGTH];

	for (uchar i = 0; i < SPONGE_LENGTH; ++i)
		sponge[i] = 0;

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
	__constant uchar const *d_message, __constant uchar const *d_target, ulong const startPosition,
	__global volatile ulong *restrict solutions, __global volatile uint *solutionCount)
{
	uchar digest[UINT256_LENGTH];

	uchar message[MESSAGE_LENGTH];
	for (uchar i = 0; i < MESSAGE_LENGTH; ++i)
		message[i] = d_message[i];

	nonce_t nonce;
	nonce.uint64_t = get_global_id(0) + startPosition;

	for (uchar i = 0; i < UINT64_LENGTH; ++i)
		message[NONCE_POSITION + i] = nonce.uint8_t[i];

	keccak256(digest, message);

	if (islessThan(digest, d_target))
	{
#ifdef cl_khr_int64_base_atomics
		uint position = atomic_inc(&solutionCount[0]);
#else
		uint position = solutionCount[0];
		++solutionCount[0];
#endif
		if (position < (MAX_SOLUTION_COUNT)) solutions[position] = nonce.uint64_t;
	}
}
