namespace rbc {

#define MAX_CELLS_NUM 100000

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

std::vector<int> extract_neighbors(std::vector<int> adjVert, int degreemax, int v) {
  std::vector<int> myneighbors;
  for (int c = 0; c < degreemax; ++c) {
    int val = adjVert[c + degreemax * v];
    if (val == -1) break;

    myneighbors.push_back(val);
  }

  return myneighbors;
}

void setup_support(int *data, int *data2, int nentries) {
  k_rbc::texAdjVert.channelDesc = cudaCreateChannelDesc<int>();
  k_rbc::texAdjVert.filterMode = cudaFilterModePoint;
  k_rbc::texAdjVert.mipmapFilterMode = cudaFilterModePoint;
  k_rbc::texAdjVert.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert, data, &k_rbc::texAdjVert.channelDesc,
		     sizeof(int) * nentries));

  k_rbc::texAdjVert2.channelDesc = cudaCreateChannelDesc<int>();
  k_rbc::texAdjVert2.filterMode = cudaFilterModePoint;
  k_rbc::texAdjVert2.mipmapFilterMode = cudaFilterModePoint;
  k_rbc::texAdjVert2.normalized = 0;

  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert2, data2,
		     &k_rbc::texAdjVert.channelDesc, sizeof(int) * nentries));
}

void setup(int* triplets, float* orig_xyzuvw) {
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
    int retval = fscanf(f, "%d %d %d %e %e %e\n", dummy + 0, dummy + 1,
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
    int retval = fscanf(f, "%d %d %d %d %d\n", dummy + 0, dummy + 1,
			      &tri.x, &tri.y, &tri.z);
    if (retval != 5) break;
    triangles.push_back(tri);
  }
  fclose(f);

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
    int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};
    for (int d = 0; d < 3; ++d) {
      adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
    }
  }

  std::vector<int> maxldeg;
  for (int i = 0; i < RBCnv; ++i)
    maxldeg.push_back(adjacentPairs[i].size());
  int degreemax = *max_element(maxldeg.begin(), maxldeg.end());
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
				    s2.end(), result.begin()) -
	result.begin();

      int myguy = result[0] == v;

      adjVert2[i + degreemax * v] = result[myguy];
    }
  }
  
  float *xyzuvw_host = new float[6 * RBCnv * sizeof(float)];
  for (int i = 0; i < RBCnv; i++) {
    xyzuvw_host[6 * i + 0] = rv[i].r[0];
    xyzuvw_host[6 * i + 1] = rv[i].r[1];
    xyzuvw_host[6 * i + 2] = rv[i].r[2];
    xyzuvw_host[6 * i + 3] = 0;
    xyzuvw_host[6 * i + 4] = 0;
    xyzuvw_host[6 * i + 5] = 0;
  }

  CC(cudaMemcpy(orig_xyzuvw, xyzuvw_host, RBCnv * 6 * sizeof(float), H2D));

  delete[] xyzuvw_host;

  float *devtrs4;
  CC(cudaMalloc(&devtrs4, RBCnt * 4 * sizeof(int)));
  CC(cudaMemcpy(devtrs4, trs4, RBCnt * 4 * sizeof(int),
		H2D));
  delete[] trs4;

  int nentries = adjVert.size();

  int *ptr, *ptr2;
  CC(cudaMalloc(&ptr, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries,
		H2D));

  CC(cudaMalloc(&ptr2, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries,
		H2D));

  setup_support(ptr, ptr2, nentries);

  k_rbc::texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
  k_rbc::texTriangles4.filterMode = cudaFilterModePoint;
  k_rbc::texTriangles4.mipmapFilterMode = cudaFilterModePoint;
  k_rbc::texTriangles4.normalized = 0;

  k_rbc::texVertices.channelDesc = cudaCreateChannelDesc<float2>();
  k_rbc::texVertices.filterMode = cudaFilterModePoint;
  k_rbc::texVertices.mipmapFilterMode = cudaFilterModePoint;
  k_rbc::texVertices.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texTriangles4, devtrs4,
		     &k_rbc::texTriangles4.channelDesc,
		     RBCnt * 4 * sizeof(int)));


  CC(cudaFuncSetCacheConfig(k_rbc::fall_kernel<RBCnv>, cudaFuncCachePreferL1));
}

void initialize(float *device_xyzuvw,
		float *transform,
		float *orig_xyzuvw) {
  int threads = 128;
  int blocks = (RBCnv + threads - 1) / threads;

  CC(cudaMemcpyToSymbol(k_rbc::A, transform, 16 * sizeof(float)));
  CC(cudaMemcpy(device_xyzuvw, orig_xyzuvw,
		6 * RBCnv * sizeof(float),
		D2D));
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

  k_rbc::fall_kernel<RBCnv><<<blocks, threads, 0>>>(nc, host_av, device_axayaz);
}

}
