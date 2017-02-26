/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer1 {
  /* `S': size; `D': data */
  int S; T *D;
  StaticDeviceBuffer1() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PART_NUM));}
  ~StaticDeviceBuffer1() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};

/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer0 {
  /* `D': data */
  T *D; int S;
  StaticDeviceBuffer0()  {mpDeviceMalloc(&D);}
  ~StaticDeviceBuffer0() {CC(cudaFree(D));}
};

