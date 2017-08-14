namespace odstr {
namespace sub {

/* pinned buffers (array of pinned buffers) */ 
template <typename T, int N=27>
struct Pbufs {
    T **dev;   /* data on device  */
    T *dp[N];  /* device pointers */
    T *hst[N]; /* data on host    */
};

struct Send {
    int **iidx;       /* helper indices (device) */
    int *iidx_[27];   /* helper indices (pinned) */
    
    Pbufs<float2> pp; /* Send particles          */
    
    int *size_dev, *strt;
    int size[27];
    dual::I size_pin;
};

struct Recv {
    Pbufs<float2> pp; /* Recv  particles          */
    
    int *strt;
    int tags[27], size[27];
};

}
}
