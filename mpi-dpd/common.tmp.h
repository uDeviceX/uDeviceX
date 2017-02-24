/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer {
  /* `S': size; `D': data */
  int S; T *D;
  StaticDeviceBuffer() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PARTICLE_NUMBER));}
  ~StaticDeviceBuffer() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};


template <typename T> struct StaticHostBuffer {
  /* `S': size; `D': data */
  int S; T D[MAX_PARTICLE_NUMBER];
  void resize(int n)  {S = n;}
};
