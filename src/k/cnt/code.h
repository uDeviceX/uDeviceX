namespace k_cnt { /* particle cell code api */

static __device__ void get(int slot, /**/ int *soluteid, int *spid) {
    CellEntry ce;
    ce.pid = fetchID(slot);
    *soluteid = ce.code.w;
    ce.code.w = 0;
    *spid = ce.pid;
}

}
