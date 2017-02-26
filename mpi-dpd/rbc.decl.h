namespace CudaRBC {
struct Params {
  float kbT, p, lmax, kp, mpow, Area0, totArea0, totVolume0, kd, ka0, kv0, gammaT,
      gammaC, sinTheta0, cosTheta0, kb, l0;
  float sint0kb, cost0kb, kbToverp;
  int nvertices, ntriangles;
};

struct Extent {
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;
};
static Params params;

float *orig_xyzuvw;
float *host_av;
float *devtrs4;

int *triplets;

float *addfrc;
int maxCells;
}
