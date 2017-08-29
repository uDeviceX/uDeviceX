namespace dev {
/* [reg]ister a particle */
static __device__ void reg(int pid, int code, /**/ int *iidx[], int size[]) {
    int entry;
    if (code > 0) {
        entry = atomicAdd(size + code, 1);
        iidx[code][entry] = pid;
    }
}
}
