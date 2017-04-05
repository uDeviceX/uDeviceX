namespace k_fsi {
  texture<float2, cudaTextureType1D> texSolventParticles;
  texture<int, cudaTextureType1D> texCellsStart, texCellsCount;
  bool firsttime = true;
  static const int NCELLS = XS * YS * ZS;
  __constant__ int packstarts_padded[27], packcount[26];
  __constant__ Particle *packstates[26];
  __constant__ Force *packresults[26];



  __global__ void interactions_3tpp(const float2 *const particles, const int np,
				    const int nsolvent, float *const acc,
				    float *const accsolvent, const float seed) {
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    const int pid = gid / 3;
    const int zplane = gid % 3;

    if (pid >= np) return;

    const float2 dst0 = __ldg(particles + 3 * pid + 0);
    const float2 dst1 = __ldg(particles + 3 * pid + 1);
    const float2 dst2 = __ldg(particles + 3 * pid + 2);
    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
      enum {
	XCELLS = XS,
	YCELLS = YS,
	ZCELLS = ZS,
	XOFFSET = XCELLS / 2,
	YOFFSET = YCELLS / 2,
	ZOFFSET = ZCELLS / 2
      };

      const int xcenter = XOFFSET + (int)floorf(dst0.x);
      const int xstart = max(0, xcenter - 1);
      const int xcount = min(XCELLS, xcenter + 2) - xstart;

      if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

      const int ycenter = YOFFSET + (int)floorf(dst0.y);

      const int zcenter = ZOFFSET + (int)floorf(dst1.x);
      const int zmy = zcenter - 1 + zplane;
      const bool zvalid = zmy >= 0 && zmy < ZCELLS;

      int count0 = 0, count1 = 0, count2 = 0;

      if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
	const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
	spidbase = tex1Dfetch(texCellsStart, cid0);
	count0 = ((cid0 + xcount == NCELLS)
		  ? nsolvent
		  : tex1Dfetch(texCellsStart, cid0 + xcount)) -
	  spidbase;
      }

      if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
	const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
	deltaspid1 = tex1Dfetch(texCellsStart, cid1);
	count1 = ((cid1 + xcount == NCELLS)
		  ? nsolvent
		  : tex1Dfetch(texCellsStart, cid1 + xcount)) -
	  deltaspid1;
      }

      if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
	const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
	deltaspid2 = tex1Dfetch(texCellsStart, cid2);
	count2 = ((cid2 + xcount == NCELLS)
		  ? nsolvent
		  : tex1Dfetch(texCellsStart, cid2 + xcount)) -
	  deltaspid2;
      }

      scan1 = count0;
      scan2 = count0 + count1;
      ncandidates = scan2 + count2;

      deltaspid1 -= scan1;
      deltaspid2 -= scan2;
    }

    float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 3
    for (int i = 0; i < ncandidates; ++i) {
      const int m1 = (int)(i >= scan1);
      const int m2 = (int)(i >= scan2);
      const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

      const int sentry = 3 * spid;
      const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
      const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
      const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

      const float myrandnr = Logistic::mean0var1(seed, pid, spid);

      // check for particle types and compute the DPD force
      float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
	pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
      float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
	vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);

      const float3 strength = compute_dpd_force_traced(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2,
						       vel1, vel2, myrandnr);

      const float xinteraction = strength.x;
      const float yinteraction = strength.y;
      const float zinteraction = strength.z;

      xforce += xinteraction;
      yforce += yinteraction;
      zforce += zinteraction;

      atomicAdd(accsolvent + sentry, -xinteraction);
      atomicAdd(accsolvent + sentry + 1, -yinteraction);
      atomicAdd(accsolvent + sentry + 2, -zinteraction);
    }

    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
  }

  void setup(const Particle *const solvent, const int npsolvent,
	     const int *const cellsstart, const int *const cellscount) {
    if (firsttime) {
      texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
      texCellsStart.filterMode = cudaFilterModePoint;
      texCellsStart.mipmapFilterMode = cudaFilterModePoint;
      texCellsStart.normalized = 0;

      texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
      texCellsCount.filterMode = cudaFilterModePoint;
      texCellsCount.mipmapFilterMode = cudaFilterModePoint;
      texCellsCount.normalized = 0;

      texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
      texSolventParticles.filterMode = cudaFilterModePoint;
      texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
      texSolventParticles.normalized = 0;

      CC(cudaFuncSetCacheConfig(interactions_3tpp, cudaFuncCachePreferL1));

      firsttime = false;
    }

    size_t textureoffset = 0;

    if (npsolvent) {
      CC(cudaBindTexture(&textureoffset, &texSolventParticles, solvent,
			 &texSolventParticles.channelDesc,
			 sizeof(float) * 6 * npsolvent));
    }

    const int ncells = XS * YS * ZS;

    CC(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart,
		       &texCellsStart.channelDesc, sizeof(int) * ncells));

    CC(cudaBindTexture(&textureoffset, &texCellsCount, cellscount,
		       &texCellsCount.channelDesc, sizeof(int) * ncells));
  }


  __global__ void interactions_halo(const int nparticles_padded,
				    const int nsolvent, float *const accsolvent,
				    const float seed) {
    const int laneid = threadIdx.x & 0x1f;
    const int warpid = threadIdx.x >> 5;
    const int localbase = 32 * (warpid + 4 * blockIdx.x);
    const int pid = localbase + laneid;

    if (localbase >= nparticles_padded) return;

    int nunpack;
    float2 dst0, dst1, dst2;
    float *dst = NULL;

    {
      const uint key9 = 9 * (localbase >= packstarts_padded[9]) +
	9 * (localbase >= packstarts_padded[18]);
      const uint key3 = 3 * (localbase >= packstarts_padded[key9 + 3]) +
	3 * (localbase >= packstarts_padded[key9 + 6]);
      const uint key1 = (localbase >= packstarts_padded[key9 + key3 + 1]) +
	(localbase >= packstarts_padded[key9 + key3 + 2]);
      const int code = key9 + key3 + key1;
      const int unpackbase = localbase - packstarts_padded[code];

      nunpack = min(32, packcount[code] - unpackbase);

      if (nunpack == 0) return;

      k_common::read_AOS6f((float2 *)(packstates[code] + unpackbase), nunpack, dst0, dst1,
		 dst2);

      dst = (float *)(packresults[code] + unpackbase);
    }

    float xforce = 0, yforce = 0, zforce = 0;

    const int nzplanes = laneid < nunpack ? 3 : 0;

    for (int zplane = 0; zplane < nzplanes; ++zplane) {
      int scan1, scan2, ncandidates, spidbase;
      int deltaspid1, deltaspid2;

      {
	enum {
	  XCELLS = XS,
	  YCELLS = YS,
	  ZCELLS = ZS,
	  XOFFSET = XCELLS / 2,
	  YOFFSET = YCELLS / 2,
	  ZOFFSET = ZCELLS / 2
	};

	const int NCELLS = XS * YS * ZS;
	const int xcenter = XOFFSET + (int)floorf(dst0.x);
	const int xstart = max(0, xcenter - 1);
	const int xcount = min(XCELLS, xcenter + 2) - xstart;

	if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) continue;

	const int ycenter = YOFFSET + (int)floorf(dst0.y);

	const int zcenter = ZOFFSET + (int)floorf(dst1.x);
	const int zmy = zcenter - 1 + zplane;
	const bool zvalid = zmy >= 0 && zmy < ZCELLS;

	int count0 = 0, count1 = 0, count2 = 0;

	if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
	  const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
	  spidbase = tex1Dfetch(texCellsStart, cid0);
	  count0 = ((cid0 + xcount == NCELLS)
		    ? nsolvent
		    : tex1Dfetch(texCellsStart, cid0 + xcount)) -
	    spidbase;
	}

	if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
	  const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
	  deltaspid1 = tex1Dfetch(texCellsStart, cid1);
	  count1 = ((cid1 + xcount == NCELLS)
		    ? nsolvent
		    : tex1Dfetch(texCellsStart, cid1 + xcount)) -
	    deltaspid1;
	}

	if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
	  const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
	  deltaspid2 = tex1Dfetch(texCellsStart, cid2);
	  count2 = ((cid2 + xcount == NCELLS)
		    ? nsolvent
		    : tex1Dfetch(texCellsStart, cid2 + xcount)) -
	    deltaspid2;
	}

	scan1 = count0;
	scan2 = count0 + count1;
	ncandidates = scan2 + count2;

	deltaspid1 -= scan1;
	deltaspid2 -= scan2;
      }

      for (int i = 0; i < ncandidates; ++i) {
	const int m1 = (int)(i >= scan1);
	const int m2 = (int)(i >= scan2);
	const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

	const int sentry = 3 * spid;
	const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
	const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
	const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

	const float myrandnr = Logistic::mean0var1(seed, pid, spid);

	// check for particle types and compute the DPD force
	float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
	  pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
	float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
	  vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
	
    const float3 strength = compute_dpd_force_traced(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2,
							 vel1, vel2, myrandnr);

	const float xinteraction = strength.x;
	const float yinteraction = strength.y;
	const float zinteraction = strength.z;

	xforce += xinteraction;
	yforce += yinteraction;
	zforce += zinteraction;

	atomicAdd(accsolvent + sentry, -xinteraction);
	atomicAdd(accsolvent + sentry + 1, -yinteraction);
	atomicAdd(accsolvent + sentry + 2, -zinteraction);
      }
    }

    k_common::write_AOS3f(dst, nunpack, xforce, yforce, zforce);
  }
}
