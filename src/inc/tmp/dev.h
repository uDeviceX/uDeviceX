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

