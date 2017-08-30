namespace dev {
#define i2d(i) { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 }
static int __device__ estimate(const int i) {
    const int d[3] = i2d(i);
    int nhalodir[3] =  {
        d[0] != 0 ? 1 : XS,
        d[1] != 0 ? 1 : YS,
        d[2] != 0 ? 1 : ZS
    };
    int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
    int safety_factor = 4;
    return numberdensity * safety_factor * nhalocells;
}

static __device__ void report(int size, int fid) {
}

static __device__ void check(int size, int fid) {
    if (fid < size) return;
    else {
        report(size, fid);
        assert(0);
    }
}

/* [reg]ister a particle */
static __device__ void reg(int pid, int fid, /**/ int *iidx[], int size[]) {
    int entry;
    if (fid > 0) {
        entry = atomicAdd(size + fid, 1);
        iidx[fid][entry] = pid;
        check(size[fid], fid);
    }
}
}
