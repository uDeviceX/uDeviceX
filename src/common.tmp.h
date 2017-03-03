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

template <typename T> struct PinnedHostBuffer1 {
private:
  int capacity;
public:
  /* `S': size; `D' is for data; `DP' device pointer */
  int S;  T *D, *DP;

  explicit PinnedHostBuffer1(int n = 0)
    : capacity(0), S(0), D(NULL), DP(NULL) {
    resize(n);
  }

  ~PinnedHostBuffer1() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }

  void resize(const int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL) CC(cudaFreeHost(D));
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }

};


template <typename T> struct PinnedHostBuffer2 {
private:
  int capacity, S;
public:
  /* `S': size; `D' is for data; `DP' device pointer */
  T *D, *DP;

  explicit PinnedHostBuffer2(int n = 0)
    : capacity(0), S(0), D(NULL), DP(NULL) {
    resize(n);
  }

  ~PinnedHostBuffer2() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }

  void resize(const int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL) CC(cudaFreeHost(D));
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }

};
