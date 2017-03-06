namespace k_sim {

__global__ void update(bool rbcflag, Particle* pp, Force* ff, int n) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  if (pid >= n) return;

  float *v = pp[pid].v, *r = pp[pid].r, *f = ff[pid].f;

  float mass = rbcflag ? rbc_mass : 1;
  lastbit::Preserver up(v[0]);
  v[0] += f[0]/mass*dt;
  v[1] += f[1]/mass*dt;
  v[2] += f[2]/mass*dt;

  r[0] += v[0]*dt; r[1] += v[1]*dt; r[2] += v[2]*dt;
}

__global__ void body_force(bool rbcflag, Particle* pp, Force* ff, int n, float driving_force) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  if (pid >= n) return;

  float *r = pp[pid].r, *f = ff[pid].f;

  float mass = rbcflag ? rbc_mass : 1;

  float dy = r[1] - glb::r0[1]; /* coordinate relative to domain
				    center */
  if (doublepoiseuille && dy <= 0) driving_force *= -1;
  f[0] += mass*driving_force;
}

 __global__ void clear_velocity(Particle *pp, int n)  {
   int pid = threadIdx.x + blockDim.x * blockIdx.x;
   if (pid >= n) return;
   lastbit::Preserver up(pp[pid].v[0]);
   for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
 }

__global__ void ic_shear_velocity(Particle *pp, int n)  {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  if (pid >= n) return;
  lastbit::Preserver up(pp[pid].v[0]);
  float z = pp[pid].r[2] - glb::r0[2];
  float vx = gamma_dot*z, vy = 0, vz = 0;
  pp[pid].v[0] = vx; pp[pid].v[1] = vy; pp[pid].v[2] = vz;
}

static __global__ void make_texture(float4 *__restrict xyzouvwoo,
				    ushort4 *__restrict xyzo_half,
				    const float *__restrict xyzuvw, const uint n) {
  extern __shared__ volatile float smem[];

  uint warpid = threadIdx.x / 32;
  uint lane = threadIdx.x % 32;

  uint i = (blockIdx.x * blockDim.x + threadIdx.x) & 0xFFFFFFE0U;

  float2 *base = (float2 *)(xyzuvw + i * 6);
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

} /* end of k_sim */
