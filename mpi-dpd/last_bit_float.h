

namespace last_bit_float {
// return a last bit of a float
// See http://stackoverflow.com/a/1723938
__host__ __device__ bool get(const float f);

// set a last bit of a float to `bit'
__host__ __device__ void set(float& f, const bool bit);

/* Last bit preserver If you do last_bit_float::Preserver zp(z);
   the last bit of `z' will be restored if variable `zp' goes out
   of the scope (see mpi-dpd/main_last_bit_float.cu) */
class Preserver {
  float& _f;
  const bool bit;
 public:
  __host__ __device__ explicit Preserver(float& f);
  __host__ __device__ ~Preserver();
};
}
