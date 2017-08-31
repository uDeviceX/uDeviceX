namespace k_cnt { /* particle cell code api */

#define NMAX (256*256*256)
static __device__ void get(int i, /**/ int *s, int *pid) {
    /* pid: particle id; s: solute id; i : slot */
    CellEntry c;
    c.pid = fetchID(i);
    *pid = c.pid % NMAX;
    *s   = c.pid / NMAX;
}

static __device__ void set(int s, int pid, int i, /**/ CellEntry *cells) {
    /* pid: particle id; s: solute id; i : slot */
    assert(pid < NMAX);
    CellEntry c;
    c.pid = NMAX * s + pid;
    cells[i] = c;
}

}

#undef NMAX
