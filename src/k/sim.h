namespace k_sim {

__global__ void update(bool rbcflag, float2 * _pdata, float * _adata,
		       int nparticles, float _driving_force) {
  int warpid = threadIdx.x >> 5;
  int base = 32 * (warpid + 4 * blockIdx.x);
  int nsrc = min(32, nparticles - base);

  float2 *pdata = _pdata + 3 * base;
  float *adata = _adata + 3 * base;

  int laneid;
  asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

  int nwords = 3 * nsrc;

  float2 s0, s1, s2;
  float ax, ay, az;

  if (laneid < nwords)
    {
      s0 = __ldg(pdata + laneid);
      ax = __ldg(adata + laneid);
    }

  if (laneid + 32 < nwords)
    {
      s1 = __ldg(pdata + laneid + 32);
      ay = __ldg(adata + laneid + 32);
    }

  if (laneid + 64 < nwords)
    {
      s2 = __ldg(pdata + laneid + 64);
      az = __ldg(adata + laneid + 64);
    }

  {
    int srclane0 = (3 * laneid + 0) & 0x1f;
    int srclane1 = (srclane0 + 1) & 0x1f;
    int srclane2 = (srclane0 + 2) & 0x1f;

    int start = laneid % 3;

    {
      float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
      float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
      float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

      s0.x = t0;
      s1.x = t1;
      s2.x = t2;
    }

    {
      float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
      float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
      float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

      s0.y = t0;
      s1.y = t1;
      s2.y = t2;
    }

    {
      float t0 = __shfl(start == 0 ? ax : start == 1 ? ay : az, srclane0);
      float t1 = __shfl(start == 0 ? az : start == 1 ? ax : ay, srclane1);
      float t2 = __shfl(start == 0 ? ay : start == 1 ? az : ax, srclane2);

      ax = t0;
      ay = t1;
      az = t2;
    }
  }

  int type; float driving_force, mass, vx = s1.y, y = s0.y;
  if      (rbcflag                             ) type = MEMB_TYPE;
  else if (!rbcflag &&  lastbit::get(vx)) type =  IN_TYPE;
  else if (!rbcflag && !lastbit::get(vx)) type = OUT_TYPE;
  mass          = (type == MEMB_TYPE) ? rbc_mass : 1;
  /* TODO: no driving force on "inner" particles */
  driving_force = (type ==   IN_TYPE) ? 0        : _driving_force;
  float y0 = glb::r0[1]; /* domain center in local coordinates */
  if (doublepoiseuille && y <= y0) driving_force *= -1;

  s1.y += (ax/mass + driving_force) * dt;
  s2.x += ay/mass * dt;
  s2.y += az/mass * dt;

  s0.x += s1.y * dt;
  s0.y += s2.x * dt;
  s1.x += s2.y * dt;

  {
    int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
    int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
    int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

    int start = laneid % 3;

    {
      float t0 = __shfl(s0.x, srclane0);
      float t1 = __shfl(s2.x, srclane1);
      float t2 = __shfl(s1.x, srclane2);

      s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
      s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
      s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    {
      float t0 = __shfl(s0.y, srclane0);
      float t1 = __shfl(s2.y, srclane1);
      float t2 = __shfl(s1.y, srclane2);

      s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
      s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
      s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    {
      float t0 = __shfl(ax, srclane0);
      float t1 = __shfl(az, srclane1);
      float t2 = __shfl(ay, srclane2);

      ax = start == 0 ? t0 : start == 1 ? t2 : t1;
      ay = start == 0 ? t1 : start == 1 ? t0 : t2;
      az = start == 0 ? t2 : start == 1 ? t1 : t0;
    }
  }

  if (laneid < nwords) {
    lastbit::Preserver up1(pdata[laneid].y);
    pdata[laneid] = s0;
  }

  if (laneid + 32 < nwords) {
    lastbit::Preserver up1(pdata[laneid + 32].y);
    pdata[laneid + 32] = s1;
  }

  if (laneid + 64 < nwords) {
    lastbit::Preserver up1(pdata[laneid + 64].y);
    pdata[laneid + 64] = s2;
  }
}

 __global__ void clear_velocity(Particle *p, int n)  {
   int pid = threadIdx.x + blockDim.x * blockIdx.x;
   if (pid >= n) return;
   lastbit::Preserver up(p[pid].v[0]);
   for(int c = 0; c < 3; ++c) p[pid].v[c] = 0;
 }

__global__ void ic_shear_velocity(Particle *p, int n)  {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  if (pid >= n) return;
  lastbit::Preserver up(p[pid].v[0]);
  float z = p[pid].r[2] - glb::r0[2];
  float vx = gamma_dot*z, vy = 0, vz = 0;
  p[pid].v[0] = vx; p[pid].v[1] = vy; p[pid].v[2] = vz;
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
