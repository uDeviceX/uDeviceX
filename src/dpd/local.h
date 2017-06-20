void forces_dpd_cuda_nohost(const float4 * const xyzouvwo, const ushort4 * const xyzo_half,
                            float * const axayaz,  const int np,
                            const int * const cellsstart, const int * const cellscount,
                            const float rc,
                            const float XL, const float YL, const float ZL,
                            const float invsqrtdt,
                            const float seed);
