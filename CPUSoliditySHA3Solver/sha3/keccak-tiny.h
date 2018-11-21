/** libkeccak-tiny
 *
 * A single-file implementation of SHA-3 and SHAKE.
 *
 * Implementor: David Leon Gil
 * License: CC0, attribution kindly requested. Blame taken too,
 * but not liability.
 */
#ifndef KECCAK_FIPS202_H
#define KECCAK_FIPS202_H

#ifdef __cplusplus
extern "C" {
#endif

#define __STDC_WANT_LIB_EXT1__ 1
#include <stdint.h>
#include <stdlib.h>

#define decshake(bits) \
  int32_t shake##bits(uint8_t*, size_t, const uint8_t*, size_t);

#define decsha3(bits) \
  int32_t sha3_##bits(uint8_t*, size_t, const uint8_t*, size_t);

#define deckeccak(bits) \
  int32_t keccak_##bits(uint8_t*, size_t, const uint8_t*, size_t);

	decshake(128)
		decshake(256)
		decsha3(224)
		decsha3(256)
		decsha3(384)
		decsha3(512)

		deckeccak(256)

#ifdef __cplusplus
}
#endif

#endif