/* TODO: in-transition structure to static allocation */
template <typename T> struct SmallDeviceBuffer0 {
  /* `S': size; `D': data */
  int S; T *D;
  explicit SmallDeviceBuffer0(int n) : S(n) {CC(cudaMalloc(&D, sizeof(T) * n));}
  ~SmallDeviceBuffer0()                     {CC(cudaFree(D));}
  void resize(int n) {S = n;}
};

/* TODO: in-transition structure to static allocation */
template <typename T> struct SmallDeviceBuffer1 {
  /* `S': size; `D': data */
  int S; T *D;
  explicit SmallDeviceBuffer1(int n) : S(n) {CC(cudaMalloc(&D, sizeof(T) * n));}
  ~SmallDeviceBuffer1()                     {CC(cudaFree(D));}
};

/* TODO: in-transition structure to static allocation */
template <typename T> struct SmallDeviceBuffer2 {
  /* `S': size; `D': data */
  T *D;
  explicit SmallDeviceBuffer2(int n) : S(n) {CC(cudaMalloc(&D, sizeof(T) * n));}
  ~SmallDeviceBuffer2()                     {CC(cudaFree(D));}
private:
  int S;
};

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
  DeviceBuffer2()  {mpDeviceMalloc(&D);}
  ~DeviceBuffer2() {CC(cudaFree(D));}
};

/* TODO: in-transition structure to static allocation */
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

template <typename T> struct PinnedHostBuffer3 {
private:
  int capacity, S;
  void resize(const int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL) CC(cudaFreeHost(D));
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }
public:
  /* `S': size; `D' is for data; `DP' device pointer */
  T *D, *DP;

  explicit PinnedHostBuffer3(int n = 0)
    : capacity(0), S(0), D(NULL), DP(NULL) {
    resize(n);
  }

  ~PinnedHostBuffer3() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }
};

template <typename T> struct PinnedHostBuffer4 {
public:
  /* `D' is for data; `DP' device pointer */
  T *D, *DP;

  explicit PinnedHostBuffer4(int n) {
    CC(cudaHostAlloc(&D, sizeof(T) * n, cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }

  ~PinnedHostBuffer4() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }
};
