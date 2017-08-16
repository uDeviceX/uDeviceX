#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

namespace d {
__inline__ __device__ int lane() {
    int id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}
}

