namespace clist {

enum {
    NMAX = (256*256*256)
};

static __device__ uint encode_id(int s, int pid) {
    /* pid: particle id; s: array id */
    assert(pid < NMAX);
    return NMAX * s + pid;
}

static __device__ void decode_id(uint c, /**/ int *s, int *pid) {
    /* pid: particle id; s: array id; c : code */
    *pid = c % NMAX;
    *s   = c / NMAX;
}

} // clist
