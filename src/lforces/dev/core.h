static __device__ void core0(uint dpid, uint spid, uint spidext) {
    forces::Fo f;
    dpd(dpid, spid, /**/ &f);
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv(dpid, 1 / 32.f);
    float* acc = info.ff + xmad(base, 96.f, off);
    atomicAdd(acc     , f.x);
    atomicAdd(acc + 32, f.y);
    atomicAdd(acc + 64, f.z);

    if (spid < spidext) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv(spid, 1 / 32.f);
        float* acc = info.ff + xmad(base, 96.f, off);
        atomicAdd(acc     , -f.x);
        atomicAdd(acc + 32, -f.y);
        atomicAdd(acc + 64, -f.z);
    }
}

static __device__ void core(uint dststart, uint pshare, uint tid, uint spidext) {
    uint dpid, spid;
    asmb::get_pair(dststart, pshare, tid, /**/ &dpid, &spid);
    core0(dpid, spid, spidext);
}
