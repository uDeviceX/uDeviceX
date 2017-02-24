/* container for the gpu particles during the simulation */
template <typename T> struct StaticDeviceBuffer {
  typedef T value_type;
  /* `C': capacity; `S': size; `D' is for data*/
  int C, S; T *D;

  explicit StaticDeviceBuffer(int n = 0) : C(0), S(0), D(NULL) { resize(n); }
  ~StaticDeviceBuffer() {
    if (D != NULL) CC(cudaFree(D));
    D = NULL;
  }

  void resize(int n) {
    S = n;
    if (C >= n) return;
    if (D != NULL) CC(cudaFree(D));
    int conservative_estimate = (int)ceil(1.1 * n);
    C = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * C));
  }
};
