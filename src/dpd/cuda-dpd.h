template<int s>
inline __device__ float viscosity_function(float x)
{
  return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline __device__ float viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline __device__ float viscosity_function<0>(float x){ return x; }

void forces_dpd_cuda_nohost(const float * const xyzuvw, const float4 * const xyzouvwo, const ushort4 * const xyzo_half,
                            float * const axayaz,  const int np,
                            const int * const cellsstart, const int * const cellscount,
                            const float rc,
                            const float XL, const float YL, const float ZL,
                            const float invsqrtdt,
                            const float seed);

void forces_dpd_cuda(const float * const xp, const float * const yp, const float * const zp,
                     const float * const xv, const float * const yv, const float * const zv,
                     float * const xa, float * const ya, float * const za,
                     const int np,
                     const float rc,
                     const float LX, const float LY, const float LZ,
                     const float invsqrtdt,
                     const float seed);
