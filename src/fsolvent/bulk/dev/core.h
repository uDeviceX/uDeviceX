static __device__ float* id2ff(uint pid) {
    uint off, base;
    float *ff;
    off  = pid & 0x0000001FU;
    base = xdiv(pid, 1 / 32.f);
    ff = info.ff + xmad(base, 96.f, off);
    return ff;
}
static __device__ void core0(uint dpid, uint spid, uint spidext) {
    forces::Fo f;
    float *ff;
    dpd(dpid, spid, /**/ &f);
    ff = id2ff(dpid);
    atomicAdd(ff     , f.x);
    atomicAdd(ff + 32, f.y);
    atomicAdd(ff + 64, f.z);

    if (spid < spidext) {
        ff = id2ff(spid);
        atomicAdd(ff     , -f.x);
        atomicAdd(ff + 32, -f.y);
        atomicAdd(ff + 64, -f.z);
    }
}

static __device__ void core(uint dststart, uint pshare, uint tid, uint spidext) {
    uint dpid, spid;
    asmb::get_pair(dststart, pshare, tid, /**/ &dpid, &spid);
    core0(dpid, spid, spidext);
}
