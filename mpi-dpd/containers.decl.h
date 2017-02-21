struct ParticleArray {
  DeviceBuffer<Particle>     pp; /* xyzuvw */
  DeviceBuffer<Acceleration> aa; /* axayaz */
};

namespace Cont {
  int (*indices)[3] = NULL;
  int ntriangles = -1;
  int nvertices = -1;
  float3 origin, globalextent;
  int    coords[3];
  MPI_Comm cartcomm;
  int nranks, rank;
  int ncells = 0;
  struct TransformedExtent {
    float com[3];
    float transform[4][4];
  };
}
