#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

// tag::int[]
_I_ uint clist_encode_id(int s, int pid)
// end::int[]
{
    /* pid: particle id; s: array id */
    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

// tag::int[]
_I_ void clist_decode_id(uint c, /**/ int *s, int *pid)
// end::int[]
{
    /* pid: particle id; s: array id; c : code */
    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
