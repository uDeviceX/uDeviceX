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
  const char* fn = "rbc.off";

  float vert[3 * RBCnv];
  off::f2vert(fn, vert);
  float *pp_h = new float[6 * RBCnv];
  for (int iv = 0, i0 = 0, i1 = 0; iv < RBCnv; iv ++) {
    float RBCscale = 1.0/rc;
    float x = vert[i0++], y = vert[i0++], z = vert[i0++];
    x *= RBCscale; y *= RBCscale; z *= RBCscale;
    pp_h[i1++] = x; pp_h[i1++] = y; pp_h[i1++] = z;
    pp_h[i1++] = 0; pp_h[i1++] = 0; pp_h[i1++] = 0;
  }
  CC(cudaMemcpy(orig_xyzuvw, pp_h, RBCnv * 6 * sizeof(float), H2D));
  delete[] pp_h;

  off::f2faces(fn, faces);
  int   *trs4 = new int  [4 * RBCnt];
  for (int ifa = 0, i0 = 0, i1 = 0; ifa < RBCnt; ifa++) {
    trs4 [i0++] = faces[i1++]; trs4[i0++] = faces[i1++]; trs4[i0++] = faces[i1++];
    trs4 [i0++] = 0;
  }

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
  CC(cudaMemcpyToSymbol(k_rbc::A, transform, 16 * sizeof(float)));
  CC(cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6 * RBCnv * sizeof(float), D2D));
  k_rbc::transformKernel<<<k_cnf(RBCnv)>>>(device_xyzuvw, RBCnv);
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
  k_rbc::areaAndVolumeKernel<<<avBlocks, avThreads>>>(host_av);
  CC(cudaPeekAtLastError());

  int degreemax = 7;
  k_rbc::fall_kernel<<<k_cnf(nc*RBCnv*degreemax)>>>(nc, host_av, device_axayaz);
}

}
