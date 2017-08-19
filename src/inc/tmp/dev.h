/* TODO: in-transition structure to static allocation */
template <typename T> struct DeviceBuffer1 {
  /* `S': size; `D': data */
  int S; T *D;
  DeviceBuffer1() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PART_NUM));}
  ~DeviceBuffer1() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};

/* TODO: in-transition structure to static allocation */
template <typename T> struct DeviceBuffer2 {
  /* `D': data */
  T *D;
  DeviceBuffer2()  {Dalloc(&D, MAX_PART_NUM);}
  ~DeviceBuffer2() {CC(cudaFree(D));}
};

