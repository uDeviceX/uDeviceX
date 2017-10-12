#include <stdio.h>
#include <math.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"

#include "rig/int.h"
#include "rig/imp.h"

namespace rig {

void alloc_quants(Quants *q) {
    q->n = q->ns = q->nps = 0;
    
    Dalloc(&q->pp ,     MAX_PART_NUM);
    Dalloc(&q->ss ,     MAX_SOLIDS);
    Dalloc(&q->rr0, 3 * MAX_PART_NUM);
    Dalloc(&q->i_pp,    MAX_PART_NUM);
    
    q->pp_hst   = new Particle[MAX_PART_NUM];
    q->ss_hst   = new Solid[MAX_SOLIDS];
    q->rr0_hst  = new float[3 * MAX_PART_NUM];
    q->i_pp_hst = new Particle[MAX_PART_NUM];

    q->ss_dmp    = new Solid[MAX_SOLIDS];
    q->ss_dmp_bb = new Solid[MAX_SOLIDS];

    sub::load_solid_mesh("mesh_solid.ply", /**/ &q->nt, &q->nv,
                         &q->htt, &q->dtt, &q->hvv, &q->dvv);
}

void free_quants(Quants *q) {
    delete[] q->pp_hst;
    delete[] q->ss_hst;
    delete[] q->rr0_hst;
    delete[] q->i_pp_hst;
    
    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    if (q->htt) delete[] q->htt;
    if (q->hvv) delete[] q->hvv;

    if (q->dtt) CC(d::Free(q->dtt));
    if (q->dvv) CC(d::Free(q->dvv));

    delete[] q->ss_dmp;
    delete[] q->ss_dmp_bb;
}

static void cpy_H2D(Quants q) {
    cH2D(q.i_pp, q.i_pp_hst, q.ns * q.nv);
    cH2D(q.ss,   q.ss_hst,   q.ns);
    cH2D(q.rr0,  q.rr0_hst,  q.nps * 3);
    cH2D(q.pp,   q.pp_hst,   q.n);
}

void gen_quants(Particle *opp, int *on, Quants *q) {
    sub::gen_from_solvent(q->nt, q->nv, q->htt, q->hvv, /**/ opp, on, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst, q->pp_hst);
    sub::gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    sub::gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(*q);
}

void strt_quants(const int id, Quants *q) {
    sub::gen_from_strt(id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    sub::gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    sub::gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(*q);
}

void set_ids(Quants q) {
    sub::set_ids(q.ns, q.ss_hst, q.ss);
}

void strt_dump_templ(const Quants q) {
    sub::strt_dump_templ(q.nps, q.rr0_hst);
}

void strt_dump(const int id, const Quants q) {
    sub::strt_dump(id, q.ns, q.ss_hst);
}

} // rig
