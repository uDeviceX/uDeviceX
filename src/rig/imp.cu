#include <mpi.h>
#include <cassert>
#include <vector>

#include <conf.h>
#include "inc/conf.h"

#include "d/ker.h"
#include "d/api.h"

#include "m.h"
#include "inc/def.h"
#include "msg.h"
#include "cc.h"
#include "mc.h"

#include "mpi/wrapper.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "texo.h"
#include "inc/mpi.type.h"
#include "rigid/int.h"

#include "mesh/collision.h"
#include "mesh/dist.h"
#include "mesh/bbox.h"

#include "io/ply.h"
#include "io/restart.h"

#include "rig/imp.h"
#include "rig/ic.h"
#include "rig/share.h"
#include "rig/ini.h"

namespace rig {
namespace sub {

void load_solid_mesh(const char *fname, Mesh *dev, Mesh *hst) {
    ply::read(fname, /**/ hst);

    dev->nv = hst->nv;
    dev->nt = hst->nt;

    CC(cudaMalloc(&dev->tt, 3 * dev->nt * sizeof(int)));
    CC(cudaMalloc(&dev->vv, 3 * dev->nv * sizeof(float)));

    cH2D(dev->tt, hst->tt, 3 * dev->nt);
    cH2D(dev->vv, hst->vv, 3 * dev->nv);
}

void gen_from_solvent(const Mesh m_hst,  /* io */ Particle *opp, int *on,
                      /* o */ int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst, Particle *pp_hst) {
    // generate models
    MSG("start solid ini");
    ic::ini("rigs-ic.txt", m_hst, /**/ ns, nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG("done solid ini");

    *n = *ns * (*nps);
}

static void pp2rr(const Particle *pp, const int n, float *rr) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c)
    rr[3*i + c] = pp[i].r[c];
}

void gen_from_strt(const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst) {
    Particle *pp = new Particle[MAX_PART_NUM];
    restart::read_pp("rig", restart::TEMPL, pp, nps);
    pp2rr(pp, *nps, rr0_hst);
    delete[] pp;

    restart::read_ss("rig", id, ss_hst, ns);
    *n = *ns * (*nps);
}

void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Solid *ss_hst, Particle *pp_hst) {
    rig::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    rig::reinit_ft_hst(ns, /**/ ss_hst);
}

void gen_ipp_hst(const Solid *ss_hst, const int ns, const Mesh m_hst, Particle *i_pp_hst) {
    rig::mesh2pp_hst(ss_hst, ns, m_hst, /**/ i_pp_hst);
}

void set_ids(const int ns, Solid *ss_hst, Solid *ss_dev) {
    ic::set_ids(ns, ss_hst);
    if (ns) cH2D(ss_dev, ss_hst, ns);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

void strt_dump_templ(const int nps, const float *rr0_hst) {
    Particle *pp = new Particle[nps];
    rr2pp(rr0_hst, nps, pp);

    restart::write_pp("rig", restart::TEMPL, pp, nps);
    
    delete[] pp;
}

void strt_dump(const int id, const int ns, const Solid *ss) {
    restart::write_ss("rig", id, ss, ns);
}

} // sub
} // rig
