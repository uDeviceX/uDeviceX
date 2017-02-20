/*** from containters.h ****/
float3 origin, globalextent;
int    coords[3];
MPI_Comm cartcomm;
int nranks, rank;
int ncells = 0;
struct ParticleArray {
  int S; /* size */
  SimpleDeviceBuffer<Particle>     pp; /* xyzuvw */
  SimpleDeviceBuffer<Acceleration> aa; /* axayaz */
};
