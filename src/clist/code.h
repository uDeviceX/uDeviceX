enum { CLIST_NMAX = (256*256*256) };

// tag::int[]
static __device__ uint clist_encode_id(int s, int pid)
// end::int[]
{
    /* pid: particle id; s: array id */
    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

// tag::int[]
static __device__ void clist_decode_id(uint c, /**/ int *s, int *pid)
// end::int[]
{
    /* pid: particle id; s: array id; c : code */
    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}
