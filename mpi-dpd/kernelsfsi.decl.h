namespace KernelsFSI {
  texture<float2, cudaTextureType1D> texSolventParticles;
  texture<int, cudaTextureType1D> texCellsStart, texCellsCount;
  bool firsttime = true;

  static const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

  __constant__ int packstarts_padded[27], packcount[26];
  __constant__ Particle *packstates[26];
  __constant__ Acceleration *packresults[26];
}
