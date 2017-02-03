#include <bitset>
#include "last_bit_float.h"

namespace last_bit_float {
  __host__ __device__ bool get(const float f) {
    unsigned char const *c = reinterpret_cast<unsigned char const*>(&f);
    return c[0] & 1;
  }

  __host__ __device__ void set(float& f, const bool bit) {
    unsigned char *c = reinterpret_cast<unsigned char *>(&f);
    if (bit) c[0] |=  1;
    else     c[0] &= ~1;
  }

  __host__ __device__ Preserver::Preserver(float& f): _f(f), bit(get(f)) { }
  __host__ __device__ Preserver::~Preserver() {
    set(_f, bit);
  }
}
