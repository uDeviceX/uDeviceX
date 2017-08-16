namespace d {
inline int lane() { return 0; }
static uint3 threadIdx, blockIdx;
static dim3  blockDim;
static int   wrapSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define blockIdx  d::blockIdx
#define wrapSize  d::wrapSize
