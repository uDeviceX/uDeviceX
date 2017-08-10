/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(cudaDeviceSynchronize())

/* test if inside device function                                 */
/* usefule for small differences in __device__ __host__ functions */
#define DEVICE_FUNC (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

/* 1D texture object template */
template<typename T>
struct Texo {
    cudaTextureObject_t to;

    __device__ __forceinline__
    const T fetch(const int i) const {return tex1Dfetch<T>(to, i);}
    
    void setup(T *data, int n) {
        cudaResourceDesc resD;
        cudaTextureDesc  texD;

        memset(&resD, 0, sizeof(resD));
        resD.resType = cudaResourceTypeLinear;
        resD.res.linear.devPtr = data;
        resD.res.linear.sizeInBytes = n * sizeof(T);
        resD.res.linear.desc = cudaCreateChannelDesc<T>();

        memset(&texD, 0, sizeof(texD));
        texD.normalizedCoords = 0;
        texD.readMode = cudaReadModeElementType;

        CC(cudaCreateTextureObject(&to, &resD, &texD, NULL));
    }

    void destroy() {CC(cudaDestroyTextureObject(to));}
};

#define D2D cudaMemcpyDeviceToDevice
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define H2H cudaMemcpyHostToHost
#define A2A cudaMemcpyDefault /* "[a]ll to [a]ll" */

#define cD2D(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), D2D))
#define cH2H(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), H2H))  /* [t]to, [f]rom */
#define cA2A(t, f, n) CC(cudaMemcpy((t), (f), (n) * sizeof((f)[0]), A2A))
#define cD2H(h, d, n) CC(cudaMemcpy((h), (d), (n) * sizeof((h)[0]), D2H))
#define cH2D(d, h, n) CC(cudaMemcpy((d), (h), (n) * sizeof((h)[0]), H2D))

/* device allocation */
#define Dalloc(d, n) CC(cudaMalloc((d), (n) * sizeof((**(d)))))
#define Dfree(d)     CC(cudaFree(d))

/* pinned memory allocation */
#define Palloc(d, n) CC(cudaHostAlloc((d), (n) * sizeof((**(d))), cudaHostAllocMapped))
#define Pfree(d)     CC(cudaFreeHost(d))

template <typename T>
void mpDeviceMalloc(T **D) {
    CC(cudaMalloc(D, sizeof(T) * MAX_PART_NUM));
}

template <typename T>
void mpHostMalloc(T **D) {
  T *p;
  p = (T*)malloc(sizeof(T) * MAX_PART_NUM);
  *D = p;
}

template <typename T> struct DeviceBuffer {
    /* `C': capacity; `S': size; `D' : data*/
    int C, S; T *D;
    explicit DeviceBuffer(int n = 0) : C(0), S(0), D(NULL) { resize(n); }
    ~DeviceBuffer() {
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

template <typename T> struct PinnedHostBuffer {
private:
    int capacity;
public:
    /* `S': size; `D' is for data; `DP' device pointer */
    int S;  T *D, *DP;
    explicit PinnedHostBuffer(int n = 0)
        : capacity(0), S(0), D(NULL), DP(NULL) {
        resize(n);
    }

    ~PinnedHostBuffer() {
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

    void preserve_resize(const int n) {
        T *old = D;
        const int oldS = S;
        S = n;
        if (capacity >= n) return;
        const int conservative_estimate = (int)ceil(1.1 * n);
        capacity = 128 * ((conservative_estimate + 129) / 128);
        D = NULL;
        CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));
        if (old != NULL) {
            CC(cudaMemcpy(D, old, sizeof(T) * oldS, H2H));
            CC(cudaFreeHost(old));
        }
        CC(cudaHostGetDevicePointer(&DP, D, 0));
    }
};
