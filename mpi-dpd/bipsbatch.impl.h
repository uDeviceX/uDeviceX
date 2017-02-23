namespace BipsBatch {
  __global__ void interaction_kernel(int ndstall, float *adst, int sizeadst) {
    BatchInfo info;

    uint code, dpid;

    {
      int gid = (threadIdx.x + blockDim.x * blockIdx.x) >> 1;

      if (gid >= start[26]) return;

      {
	int key9 = 9 * ((gid >= start[9]) + (gid >= start[18]));
	int key3 = 3 * ((gid >= start[key9 + 3]) + (gid >= start[key9 + 6]));
	int key1 =
          (gid >= start[key9 + key3 + 1]) + (gid >= start[key9 + key3 + 2]);

	code = key9 + key3 + key1;
	dpid = gid - start[code];
      }

      info = batchinfos[code];

      if (dpid >= info.ndst) return;
    }

    float xp = info.xdst[0 + dpid * 6];
    float yp = info.xdst[1 + dpid * 6];
    float zp = info.xdst[2 + dpid * 6];

    float up = info.xdst[3 + dpid * 6];
    float vp = info.xdst[4 + dpid * 6];
    float wp = info.xdst[5 + dpid * 6];

    int dstbase = 3 * info.scattered_entries[dpid];

    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1 = 0, deltaspid2 = 0;
    {
      int basecid = 0, xstencilsize = 1, ystencilsize = 1, zstencilsize = 1;

      {
	if (info.dz == 0) {
	  int zcid = (int)(zp + ZSIZE_SUBDOMAIN / 2);
	  int zbasecid = max(0, -1 + zcid);
	  basecid = zbasecid;
	  zstencilsize = min(info.zcells, zcid + 2) - zbasecid;
	}

	basecid *= info.ycells;

	if (info.dy == 0) {
	  int ycid = (int)(yp + YSIZE_SUBDOMAIN / 2);
	  int ybasecid = max(0, -1 + ycid);
	  basecid += ybasecid;
	  ystencilsize = min(info.ycells, ycid + 2) - ybasecid;
	}

	basecid *= info.xcells;

	if (info.dx == 0) {
	  int xcid = (int)(xp + XSIZE_SUBDOMAIN / 2);
	  int xbasecid = max(0, -1 + xcid);
	  basecid += xbasecid;
	  xstencilsize = min(info.xcells, xcid + 2) - xbasecid;
	}

	xp -= info.dx * XSIZE_SUBDOMAIN;
	yp -= info.dy * YSIZE_SUBDOMAIN;
	zp -= info.dz * ZSIZE_SUBDOMAIN;
      }

      int rowstencilsize = 1, colstencilsize = 1, ncols = 1;

      if (info.halotype == HALO_FACE) {
	rowstencilsize = info.dz ? ystencilsize : zstencilsize;
	colstencilsize = info.dx ? ystencilsize : xstencilsize;
	ncols = info.dx ? info.ycells : info.xcells;
      } else if (info.halotype == HALO_EDGE)
	colstencilsize = max(xstencilsize, max(ystencilsize, zstencilsize));

      spidbase = __ldg(info.cellstarts + basecid);
      int count0 = __ldg(info.cellstarts + basecid + colstencilsize) - spidbase;

      int count1 = 0, count2 = 0;

      if (rowstencilsize > 1) {
	deltaspid1 = __ldg(info.cellstarts + basecid + ncols);
	count1 = __ldg(info.cellstarts + basecid + ncols + colstencilsize) -
	  deltaspid1;
      }

      if (rowstencilsize > 2) {
	deltaspid2 = __ldg(info.cellstarts + basecid + 2 * ncols);
	count2 = __ldg(info.cellstarts + basecid + 2 * ncols + colstencilsize) -
	  deltaspid2;
      }

      scan1 = count0;
      scan2 = scan1 + count1;
      ncandidates = scan2 + count2;

      deltaspid1 -= scan1;
      deltaspid2 -= scan2;
    }

    float2 *xsrc = info.xsrc;
    int mask = info.mask;
    float seed = info.seed;

    float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 2
    for (uint i = threadIdx.x & 1; i < ncandidates; i += 2) {
      int m1 = (int)(i >= scan1);
      int m2 = (int)(i >= scan2);
      uint spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

      float2 s0 = __ldg(xsrc + 0 + spid * 3);
      float2 s1 = __ldg(xsrc + 1 + spid * 3);
      float2 s2 = __ldg(xsrc + 2 + spid * 3);

      uint arg1 = mask ? dpid : spid;
      uint arg2 = mask ? spid : dpid;
      float myrandnr = Logistic::mean0var1(seed, arg1, arg2);

      // check for particle types and compute the DPD force
      float3 pos1 = make_float3(xp, yp, zp), pos2 = make_float3(s0.x, s0.y, s1.x);
      float3 vel1 = make_float3(up, vp, wp), vel2 = make_float3(s1.y, s2.x, s2.y);
      int type1 = last_bit_float::get(vel1.x) ? IN_TYPE : OUT_TYPE;
      int type2 = last_bit_float::get(vel2.x) ? IN_TYPE : OUT_TYPE;
      float3 strength = compute_dpd_force_traced(type1, type2, pos1, pos2, vel1,
						 vel2, myrandnr);

      xforce += strength.x;
      yforce += strength.y;
      zforce += strength.z;
    }

    atomicAdd(adst + dstbase + 0, xforce);
    atomicAdd(adst + dstbase + 1, yforce);
    atomicAdd(adst + dstbase + 2, zforce);
  }

  void interactions(float invsqrtdt, BatchInfo infos[20],
		    float *acc, int n) {
    if (firstcall) {
      CC(cudaEventCreate(&evhalodone, cudaEventDisableTiming));
      CC(cudaFuncSetCacheConfig(interaction_kernel, cudaFuncCachePreferL1));
      firstcall = false;
    }

    CC(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 26, 0,
			       cudaMemcpyHostToDevice));

    static unsigned int hstart_padded[27];

    hstart_padded[0] = 0;
    for (int i = 0; i < 26; ++i)
      hstart_padded[i + 1] =
        hstart_padded[i] + 16 * (((unsigned int)infos[i].ndst + 15) / 16);

    CC(cudaMemcpyToSymbolAsync(start, hstart_padded, sizeof(hstart_padded), 0,
			       cudaMemcpyHostToDevice));

    int nthreads = 2 * hstart_padded[26];

    CC(cudaEventRecord(evhalodone));

    if (nthreads)
      interaction_kernel<<<(nthreads + 127) / 128, 128, 0>>>
	(nthreads, acc, n);
  }

}
