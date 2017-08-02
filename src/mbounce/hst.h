namespace mbounce {
namespace hst {

void bounce_tcells(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                   const int n, /**/ Particle *pp, Solid *ss) {
#ifdef debug_output
    if (dstep % part_freq == 0)
        for (int c = 0; c < NBBSTATES; ++c) bbstates_hst[c] = 0;
#endif

    for (int i = 0; i < n; ++i) {
        const Particle p1 = pp[i];

        Particle p0; rvprev(p1.r, p1.v, ff[i].f, /**/ p0.r, p0.v);

        const int xcid_ = int (p1.r[X] + XS/2);
        const int ycid_ = int (p1.r[Y] + YS/2);
        const int zcid_ = int (p1.r[Z] + ZS/2);

        float h = 2*dt; // must be higher than any valid result
        float rw[3], vw[3];

        int sid = -1;

        for (int zcid = max(zcid_-1, 0); zcid <= min(zcid_ + 1, ZS - 1); ++zcid)
            for (int ycid = max(ycid_-1, 0); ycid <= min(ycid_ + 1, YS - 1); ++ycid)
                for (int xcid = max(xcid_-1, 0); xcid <= min(xcid_ + 1, XS - 1); ++xcid) {
                    const int cid = xcid + XS * (ycid + YS * zcid);
                    const int start = tcellstarts[cid];
                    const int count = tcellcounts[cid];

                    for (int j = start; j < start + count; ++j) {
                        const int tid = tids[j];
                        const int it  = tid % m.nt;
                        const int mid = tid / m.nt;

                        if (find_better_intersection(m.tt, it, i_pp + mid * m.nv, &p0, /*io*/ &h, /**/ rw, vw))
                            sid = mid;
                    }
                }

        if (sid != -1) {
            Particle pn;
            bounce_back(&p0, rw, vw, h, /**/ &pn);

            float dP[3], dL[3];
            lin_mom_solid(p1.v, pn.v, /**/ dP);
            ang_mom_solid(ss[sid].com, rw, p0.v, pn.v, /**/ dL);

            pp[i] = pn;

            ss[sid].fo[X] += dP[X];
            ss[sid].fo[Y] += dP[Y];
            ss[sid].fo[Z] += dP[Z];

            ss[sid].to[X] += dL[X];
            ss[sid].to[Y] += dL[Y];
            ss[sid].to[Z] += dL[Z];
        }
    }

#ifdef debug_output
    if ((++dstep) % part_freq == 0)
        print_states(bbstates_hst);
#endif
}


} // hst
} // mbounce
