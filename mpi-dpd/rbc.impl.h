namespace rbc {

void eat_until(FILE *f, std::string target) {
  while (!feof(f)) {
    char buf[2048];
    fgets(buf, 2048, f);

    if (std::string(buf) == target) {
      fgets(buf, 2048, f);
      break;
    }
  }
}

std::vector<int> extract_neighbors(std::vector<int> adjVert,
				   const int degreemax, const int v) {
  std::vector<int> myneighbors;
  for (int c = 0; c < degreemax; ++c) {
    const int val = adjVert[c + degreemax * v];
    if (val == -1) break;

    myneighbors.push_back(val);
  }

  return myneighbors;
}

void setup_support(const int *data, const int *data2, const int nentries) {
  texAdjVert.channelDesc = cudaCreateChannelDesc<int>();
  texAdjVert.filterMode = cudaFilterModePoint;
  texAdjVert.mipmapFilterMode = cudaFilterModePoint;
  texAdjVert.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texAdjVert, data, &texAdjVert.channelDesc,
		     sizeof(int) * nentries));

  texAdjVert2.channelDesc = cudaCreateChannelDesc<int>();
  texAdjVert2.filterMode = cudaFilterModePoint;
  texAdjVert2.mipmapFilterMode = cudaFilterModePoint;
  texAdjVert2.normalized = 0;

  CC(cudaBindTexture(&textureoffset, &texAdjVert2, data2,
		     &texAdjVert.channelDesc, sizeof(int) * nentries));
}

void setup() {
  FILE *f = fopen("rbc.dat", "r");
  if (!f) {
    printf("Error in cuda-rbc: data file not found!\n");
    exit(1);
  }

  eat_until(f, "Atoms\n");

  std::vector<Particle> rv;
  while (!feof(f)) {
    Particle p = {0, 0, 0, 0, 0, 0};
    int dummy[3];
    const int retval = fscanf(f, "%d %d %d %e %e %e\n", dummy + 0, dummy + 1,
			      dummy + 2, p.r, p.r + 1, p.r + 2);
    float RBCscale = 1.0/rc;
    p.r[0] *= RBCscale; p.r[1] *= RBCscale; p.r[2] *= RBCscale;
    if (retval != 6) break;
    rv.push_back(p);
  }

  eat_until(f, "Angles\n");

  std::vector<int3> triangles;

  while (!feof(f)) {
    int dummy[2];
    int3 tri;
    const int retval = fscanf(f, "%d %d %d %d %d\n", dummy + 0, dummy + 1,
			      &tri.x, &tri.y, &tri.z);
    if (retval != 5) break;
    triangles.push_back(tri);
  }
  fclose(f);

  triplets = new int[3 * triangles.size()];
  int *trs4 = new int[4 * triangles.size()];
  for (int i = 0; i < triangles.size(); i++) {
    int3 tri = triangles[i];
    triplets[3 * i + 0] = tri.x;
    triplets[3 * i + 1] = tri.y;
    triplets[3 * i + 2] = tri.z;

    trs4[4 * i + 0] = tri.x;
    trs4[4 * i + 1] = tri.y;
    trs4[4 * i + 2] = tri.z;
    trs4[4 * i + 3] = 0;
  }

  std::vector<std::map<int, int> > adjacentPairs(RBCnv);

  for (int i = 0; i < triangles.size(); ++i) {
    const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};
    for (int d = 0; d < 3; ++d) {
      adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
    }
  }

  std::vector<int> maxldeg;
  for (int i = 0; i < RBCnv; ++i)
    maxldeg.push_back(adjacentPairs[i].size());
  const int degreemax = *max_element(maxldeg.begin(), maxldeg.end());
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

      const int nterms = set_intersection(s1.begin(), s1.end(), s2.begin(),
					  s2.end(), result.begin()) -
			 result.begin();

      const int myguy = result[0] == v;

      adjVert2[i + degreemax * v] = result[myguy];
    }
  }

  // Find stretching points
  float stretchingForce = 0;
  std::vector<std::pair<float, int> > tmp(RBCnv);
  for (int i = 0; i < RBCnv; i++) {
    tmp[i].first = rv[i].r[0];
    tmp[i].second = i;
  }
  sort(tmp.begin(), tmp.end());

  float hAddfrc[RBCnv];
  memset(hAddfrc, 0, RBCnv * sizeof(float));
  const int strVerts = 3; // 10
  for (int i = 0; i < strVerts; i++) {
    hAddfrc[tmp[i].second] = -stretchingForce / strVerts;
    hAddfrc[tmp[RBCnv - 1 - i].second] = +stretchingForce / strVerts;
  }

  CC(cudaMalloc(&addfrc, RBCnv * sizeof(float)));
  CC(cudaMemcpy(addfrc, hAddfrc, RBCnv * sizeof(float), H2D));

  float *xyzuvw_host = new float[6 * RBCnv * sizeof(float)];
  for (int i = 0; i < RBCnv; i++) {
    xyzuvw_host[6 * i + 0] = rv[i].r[0];
    xyzuvw_host[6 * i + 1] = rv[i].r[1];
    xyzuvw_host[6 * i + 2] = rv[i].r[2];
    xyzuvw_host[6 * i + 3] = 0;
    xyzuvw_host[6 * i + 4] = 0;
    xyzuvw_host[6 * i + 5] = 0;
  }

  CC(cudaMalloc(&orig_xyzuvw, RBCnv * 6 * sizeof(float)));
  CC(cudaMemcpy(orig_xyzuvw, xyzuvw_host, RBCnv * 6 * sizeof(float), H2D));

  delete[] xyzuvw_host;

  float *devtrs4;
  CC(cudaMalloc(&devtrs4, RBCnt * 4 * sizeof(int)));
  CC(cudaMemcpy(devtrs4, trs4, RBCnt * 4 * sizeof(int),
		H2D));
  delete[] trs4;

  const int nentries = adjVert.size();

  int *ptr, *ptr2;
  CC(cudaMalloc(&ptr, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries,
		H2D));

  CC(cudaMalloc(&ptr2, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries,
		H2D));

  setup_support(ptr, ptr2, nentries);

  texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
  texTriangles4.filterMode = cudaFilterModePoint;
  texTriangles4.mipmapFilterMode = cudaFilterModePoint;
  texTriangles4.normalized = 0;

  texVertices.channelDesc = cudaCreateChannelDesc<float2>();
  texVertices.filterMode = cudaFilterModePoint;
  texVertices.mipmapFilterMode = cudaFilterModePoint;
  texVertices.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texTriangles4, devtrs4,
		     &texTriangles4.channelDesc,
		     RBCnt * 4 * sizeof(int)));

  maxCells = 0;
  CC(cudaMalloc(&host_av, 1 * 2 * sizeof(float)));
  CC(cudaFuncSetCacheConfig(fall_kernel<RBCnv>, cudaFuncCachePreferL1));
}

void initialize(float *device_xyzuvw, const float (*transform)[4]) {
  const int threads = 128;
  const int blocks = (RBCnv + threads - 1) / threads;

  CC(cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)));
  CC(cudaMemcpy(device_xyzuvw, orig_xyzuvw,
		6 * RBCnv * sizeof(float),
		D2D));
  transformKernel<<<blocks, threads>>>(device_xyzuvw, RBCnv);
  CC(cudaPeekAtLastError());
}

void forces_nohost(int nc, const float *const device_xyzuvw,
		   float *const device_axayaz) {
  if (nc == 0) return;

  if (nc > maxCells) {
    maxCells = 2 * nc;
    CC(cudaFree(host_av));
    CC(cudaMalloc(&host_av, maxCells * 2 * sizeof(float)));
  }

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texVertices,
		     (float2 *)device_xyzuvw,
		     &texVertices.channelDesc,
		     nc * RBCnv * sizeof(float) * 6));

  dim3 avThreads(256, 1);
  dim3 avBlocks(1, nc);

  CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
  areaAndVolumeKernel<<<avBlocks, avThreads, 0>>>(host_av);
  CC(cudaPeekAtLastError());

  int threads = 128;
  int blocks = (nc * RBCnv * 7 + threads - 1) / threads;

  fall_kernel<RBCnv><<<blocks, threads, 0>>>(nc, host_av, device_axayaz);
  addKernel<<<(RBCnv + 127) / 128, 128, 0>>>(device_axayaz, addfrc,
							RBCnv);
}

void get_triangle_indexing(int (*&host_triplets_ptr)[3]) {
  host_triplets_ptr = (int(*)[3])triplets;
}

}
