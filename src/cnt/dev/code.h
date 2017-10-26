#define NMAX (256*256*256)
static __device__ void set(int s, int pid, int i, /**/ uint *cells) {
    /* pid: particle id; s: solute id; i : slot */
    assert(pid < NMAX);
    cells[i] = NMAX * s + pid;
}
#undef NMAX
