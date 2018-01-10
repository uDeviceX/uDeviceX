enum { CLIST_NMAX = (256*256*256) };

static __device__ uint clist_encode_id(int s, int pid) {
    /* pid: particle id; s: array id */
    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

static __device__ void clist_decode_id(uint c, /**/ int *s, int *pid) {
    /* pid: particle id; s: array id; c : code */
    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}
