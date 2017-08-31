namespace k_cnt { /* particle cell code api */

#define NMAX (256*256*256)
static __device__ void get(int i, /**/ int *s, int *pid) {
    /* pid: particle id; s: solute id; i : slot */
    CellEntry c;
    c.pid = fetchID(i);
    int pid0 = c.pid % NMAX, s0   = c.pid / NMAX;

    *s = c.code.w;
    c.code.w = 0;
    *pid = c.pid;

    assert(*pid == pid0);
    assert(  *s ==   s0);
}

static __device__ void set(int s, int pid, int i, /**/ CellEntry *cells) {
    /* pid: particle id; s: solute id; i : slot */
    assert(pid < NMAX);
    //    int c0 = NMAX * s + pid;
    //    assert( c.pid == c0);

    CellEntry c;
    c.pid = pid;
    c.code.w = s;
    cells[i] = c;
}

}

#undef NMAX
