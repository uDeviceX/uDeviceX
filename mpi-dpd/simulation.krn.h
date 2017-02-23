static __global__ void make_texture(float4 *__restrict xyzouvwoo,
			     ushort4 *__restrict xyzo_half,
			     const float *__restrict xyzuvw, const uint n) {
  extern __shared__ volatile float smem[];
  const uint warpid = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;

  const uint i = (blockIdx.x * blockDim.x + threadIdx.x) & 0xFFFFFFE0U;

  const float2 *base = (float2 *)(xyzuvw + i * 6);
#pragma unroll 3
  for (uint j = lane; j < 96; j += 32) {
    float2 u = base[j];
    // NVCC bug: no operator = between volatile float2 and float2
    asm volatile("st.volatile.shared.v2.f32 [%0], {%1, %2};"
		 :
		 : "r"((warpid * 96 + j) * 8), "f"(u.x), "f"(u.y)
		 : "memory");
  }
  // SMEM: XYZUVW XYZUVW ...
  uint pid = lane / 2;
  const uint x_or_v = (lane % 2) * 3;
  xyzouvwoo[i * 2 + lane] =
      make_float4(smem[warpid * 192 + pid * 6 + x_or_v + 0],
		  smem[warpid * 192 + pid * 6 + x_or_v + 1],
		  smem[warpid * 192 + pid * 6 + x_or_v + 2], 0);
  pid += 16;
  xyzouvwoo[i * 2 + lane + 32] =
      make_float4(smem[warpid * 192 + pid * 6 + x_or_v + 0],
		  smem[warpid * 192 + pid * 6 + x_or_v + 1],
		  smem[warpid * 192 + pid * 6 + x_or_v + 2], 0);

  xyzo_half[i + lane] =
      make_ushort4(__float2half_rn(smem[warpid * 192 + lane * 6 + 0]),
		   __float2half_rn(smem[warpid * 192 + lane * 6 + 1]),
		   __float2half_rn(smem[warpid * 192 + lane * 6 + 2]), 0);
}
