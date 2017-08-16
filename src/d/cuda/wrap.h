namespace d {
__inline__ __device__ int lane() {
    int id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}
}
