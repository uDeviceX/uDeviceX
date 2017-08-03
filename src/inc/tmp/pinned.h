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

template <typename T> struct PinnedHostBuffer5 {
public:
    /* `D' is for data */
    T *D;

    explicit PinnedHostBuffer5(int n) {
        CC(cudaHostAlloc(&D, sizeof(T) * n, cudaHostAllocMapped));
    }

    ~PinnedHostBuffer5() {
        if (D != NULL) CC(cudaFreeHost(D));
        D = NULL;
    }
};
