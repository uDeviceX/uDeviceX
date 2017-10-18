namespace clist {

#define NMAX (256*256*256)
static __device__ int encode_id(int s, int pid) {
    /* pid: particle id; s: array id */
    assert(pid < NMAX);
    return NMAX * s + pid;
}

static __device__ void decode_id(int c, /**/ int *s, int *pid) {
    /* pid: particle id; s: array id; c : code */
    *pid = c % NMAX;
    *s   = c / NMAX;
}
#undef NMAX

} // clist
