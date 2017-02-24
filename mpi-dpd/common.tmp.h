#define MAX_PARTICLE_NUMBER 5000000

/* container for the gpu particles during the simulation */
template <typename T> struct StaticDeviceBuffer {
  /* `S': size; `D' is for data*/
  int S; T *D;
  StaticDeviceBuffer() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PARTICLE_NUMBER));}
  ~StaticDeviceBuffer() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};
