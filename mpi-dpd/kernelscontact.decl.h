namespace KernelsContact {
  static const int maxsolutes = 32;
  enum {
    XCELLS = XSIZE_SUBDOMAIN,
    YCELLS = YSIZE_SUBDOMAIN,
    ZCELLS = ZSIZE_SUBDOMAIN,
    XOFFSET = XCELLS / 2,
    YOFFSET = YCELLS / 2,
    ZOFFSET = ZCELLS / 2
  };
  static const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;
  union CellEntry {
    int pid;
    uchar4 code;
  };
  texture<int, cudaTextureType1D> texCellsStart, texCellEntries;
  __constant__ int cnsolutes[maxsolutes];
  __constant__ const float2 *csolutes[maxsolutes];
  __constant__ float *csolutesacc[maxsolutes];
}
