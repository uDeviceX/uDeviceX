#include "glb.h"

namespace glb {
  /* global variables visible for every kernel */
  __constant__ float xc[3];

  void sim() {
    float xc0[3] = {1, 2, 3};
    cudaMemcpyToSymbol(xc, xc0, sizeof(xc0));
  }
}
