/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer {
  /* `S': size; `D': data */
  int S; T *D;
  StaticDeviceBuffer() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PART_NUM));}
  ~StaticDeviceBuffer() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};

template <typename T> struct StaticHostBuffer {
  /* `D': data */
  T D[MAX_PART_NUM];
};
