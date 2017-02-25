/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer {
  /* `S': size; `D': data */
  int S; T *D;
  StaticDeviceBuffer() : S(0) {CC(cudaMalloc(&D, sizeof(T) * MAX_PART_NUM));}
  ~StaticDeviceBuffer() {      CC(cudaFree(D));}
  void resize(int n) {S = n;}
};

template <typename T>
void mpDeviceMalloc(T **D) { /* a "[m]ax [p]article number" device
			       allocation (takes a pointer on
			       pointer!) */
  CC(cudaMalloc(D, sizeof(T) * MAX_PART_NUM));
}

/* TODO: in-transition structure to static allocation */
template <typename T> struct StaticDeviceBuffer0 {
  /* `D': data */
  T *D;
  StaticDeviceBuffer0()  {mpDeviceMalloc(&D);}
  ~StaticDeviceBuffer0() {CC(cudaFree(D));}
};

