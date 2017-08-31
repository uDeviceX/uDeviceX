namespace k_cnt { /* particle cell code api */

static __device__ void get(int i, /**/ int *s, int *spid) {
    CellEntry c;
    c.pid = fetchID(i);
    *s = c.code.w;
    c.code.w = 0;
    *spid = c.pid;
}

static __device__ void set(int s, int pid, int i, /**/ CellEntry *cells) {
    CellEntry c;
    c.pid = pid;
    c.code.w = s;
    cells[i] = c;
}

}
