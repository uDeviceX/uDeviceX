namespace dev { /* particle cell code api */

#define NMAX (256*256*256)
static __device__ void get(int i, /**/ int *s, int *pid) {
    /* pid: particle id; s: solute id; i : slot */
    int c;
    c    = fetchID(i);
    *pid = c % NMAX;
    *s   = c / NMAX;
}

static __device__ void set(int s, int pid, int i, /**/ int *cells) {
    /* pid: particle id; s: solute id; i : slot */
    assert(pid < NMAX);
    cells[i] = NMAX * s + pid;
}

}

#undef NMAX
