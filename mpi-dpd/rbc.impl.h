namespace rbc {

#define MAX_CELLS_NUM 100000

std::vector<int> extract_neighbors(std::vector<int> adjVert, int degreemax, int v) {
  std::vector<int> myneighbors;
  for (int c = 0; c < degreemax; ++c) {
    int val = adjVert[c + degreemax * v];
    if (val == -1) break;
    myneighbors.push_back(val);
  }
  return myneighbors;
}

#define setup_texture(T, TYPE) do {		     \
    (T).channelDesc = cudaCreateChannelDesc<TYPE>(); \
    (T).filterMode = cudaFilterModePoint;	     \
    (T).mipmapFilterMode = cudaFilterModePoint;	     \
    (T).normalized = 0;				     \
} while (false)

void setup_support(int *data, int *data2, int nentries) {
  setup_texture(k_rbc::texAdjVert, int);

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert, data,
		     &k_rbc::texAdjVert.channelDesc, sizeof(int) * nentries));

  setup_texture(k_rbc::texAdjVert2, int);
  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert2, data2,
		     &k_rbc::texAdjVert.channelDesc, sizeof(int) * nentries));
}

void setup(int* faces, float* orig_xyzuvw) {
  char buf[1024];
  FILE *f = fopen("rbc.off", "r");
  fgets(buf, sizeof buf, f); /* skip OFF */
  int nv, ne, nf;
  fscanf(f, "%d %d %d", &nv, &nf, &ne);
  if (nv != RBCnv) {
    printf("(rbc.impl.h) nv = %d and RBCnv = %d\n", nv, RBCnv); exit(1);
  };
  if (nf != RBCnt) {
    printf("(rbc.impl.h) nf = %d and RBCnt = %d\n", nf, RBCnt); exit(1);
  };

  float *pp_h = new float[6 * RBCnv * sizeof(float)];
  for (int iv = 0; iv < RBCnv; iv ++) {
    float x, y, z;
    fscanf(f, "%e %e %e", &x, &y, &z);
    float RBCscale = 1.0/rc;
    x *= RBCscale; y *= RBCscale; z *= RBCscale;
    int ib = 6*iv; pp_h[ib++] = x; pp_h[ib++] = y; pp_h[ib++] = z;
		   pp_h[ib++] = 0; pp_h[ib++] = 0; pp_h[ib++] = 0;
  }
  CC(cudaMemcpy(orig_xyzuvw, pp_h, RBCnv * 6 * sizeof(float), H2D));
  delete[] pp_h;

  int   *trs4 = new int  [4 * RBCnt];
  for (int ifa = 0; ifa < RBCnt; ifa++) {
    int f0, f1, f2, ib;
    fscanf(f, "%*d %d %d %d", &f0, &f1, &f2);
    ib = 3*ifa; faces[ib++] = f0; faces[ib++] = f1; faces[ib++] = f2;
    ib = 4*ifa; trs4 [ib++] = f0; trs4 [ib++] = f1; trs4 [ib++] = f2;
		trs4 [ib++] =  0;
  }
  fclose(f);
  float *devtrs4;
  CC(cudaMalloc(&devtrs4,       RBCnt * 4 * sizeof(int)));
  CC(cudaMemcpy( devtrs4, trs4, RBCnt * 4 * sizeof(int), H2D));
  delete[] trs4;

  std::vector<std::map<int, int> > adjacentPairs(RBCnv);
  for (int ifa = 0; ifa < RBCnt; ifa++) {
    int ib = 3*ifa;
    int f0 = faces[ib++], f1 = faces[ib++], f2 = faces[ib++];
    adjacentPairs[f0][f1] = f2;
    adjacentPairs[f2][f0] = f1;
    adjacentPairs[f1][f2] = f0;
  }

  int degreemax = 0;
  for (int i = 0; i < RBCnv; i++) {
    int d = adjacentPairs[i].size();
    if (d > degreemax) degreemax = d;
  }

  std::vector<int> adjVert(RBCnv * degreemax, -1);
  for (int v = 0; v < RBCnv; ++v) {
    std::map<int, int> l = adjacentPairs[v];
    adjVert[0 + degreemax * v] = l.begin()->first;
    int last = adjVert[1 + degreemax * v] = l.begin()->second;
    for (int i = 2; i < l.size(); ++i) {
      int tmp = adjVert[i + degreemax * v] = l.find(last)->second;
      last = tmp;
    }
  }

  std::vector<int> adjVert2(degreemax * RBCnv, -1);
  for (int v = 0; v < RBCnv; ++v) {
    std::vector<int> myneighbors = extract_neighbors(adjVert, degreemax, v);
    for (int i = 0; i < myneighbors.size(); ++i) {
      std::vector<int> s1 =
	  extract_neighbors(adjVert, degreemax, myneighbors[i]);
      std::sort(s1.begin(), s1.end());
      std::vector<int> s2 = extract_neighbors(
	  adjVert, degreemax, myneighbors[(i + 1) % myneighbors.size()]);
      std::sort(s2.begin(), s2.end());
      std::vector<int> result(s1.size() + s2.size());
      int nterms = set_intersection(s1.begin(), s1.end(), s2.begin(),
				    s2.end(), result.begin()) - result.begin();
      int myguy = result[0] == v;
      adjVert2[i + degreemax * v] = result[myguy];
    }
  }

  int nentries = adjVert.size();
  int *ptr, *ptr2;
  CC(cudaMalloc(&ptr, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries, H2D));

  CC(cudaMalloc(&ptr2, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries, H2D));

  setup_support(ptr, ptr2, nentries);

  setup_texture(k_rbc::texTriangles4, int4);
  setup_texture(k_rbc::texVertices, float2);

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texTriangles4, devtrs4,
		     &k_rbc::texTriangles4.channelDesc,
		     RBCnt * 4 * sizeof(int)));

  CC(cudaFuncSetCacheConfig(k_rbc::fall_kernel, cudaFuncCachePreferL1));
}

void initialize(float *device_xyzuvw,
		float *transform,
		float *orig_xyzuvw) {
  int threads = 128;
  int blocks = (RBCnv + threads - 1) / threads;

  CC(cudaMemcpyToSymbol(k_rbc::A, transform, 16 * sizeof(float)));
  CC(cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6 * RBCnv * sizeof(float), D2D));
  k_rbc::transformKernel<<<blocks, threads>>>(device_xyzuvw, RBCnv);
  CC(cudaPeekAtLastError());
}

void forces_nohost(int nc, float *device_xyzuvw,
		   float *device_axayaz, float* host_av) {
  if (nc == 0) return;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texVertices,
		     (float2 *)device_xyzuvw,
		     &k_rbc::texVertices.channelDesc,
		     nc * RBCnv * sizeof(float) * 6));

  dim3 avThreads(256, 1);
  dim3 avBlocks(1, nc);

  CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
  k_rbc::areaAndVolumeKernel<<<avBlocks, avThreads, 0>>>(host_av);
  CC(cudaPeekAtLastError());

  int threads = 128;
  int blocks = (nc * RBCnv * 7 + threads - 1) / threads;

  k_rbc::fall_kernel<<<blocks, threads, 0>>>(nc, host_av, device_axayaz);
}

}
