#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "d/api.h"
#include "mpi/wrapper.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "imp.h"

void ini(MPI_Comm comm, int3 L, float3 vtarget, float factor, /**/ PidVCont *c) {
    int ncells, nchunks;
    c->L = L;
    c->target = vtarget;
    c->Kp = 2;
    c->Ki = 1;
    c->Kd = 8;
    c->nsamples = 0;

    MC(m::Comm_dup(comm, &c->comm));

    ncells = L.x * L.y * L.z;
    CC(d::Malloc((void **) &c->gridvel, ncells * sizeof(float3)));

    nchunks = ceiln(ncells, 32);
    
    CC(d::alloc_pinned((void **) &c->avgvel, nchunks * sizeof(float3)));
    CC(d::HostGetDevicePointer((void **) &c->davgvel, c->avgvel, 0));

    c->f = c->sume = make_float3(0, 0, 0);
    c->olde = vtarget;
}

void fin(/**/ PidVCont *c) {
    CC(d::Free(c->gridvel));
    CC(d::FreeHost(c->avgvel));
    MC(m::Comm_free(&c->comm));
}

void sample(int n, const Particle *pp, /**/ PidVCont *c) {

    ++c->nsamples;
}

float3 adjustF(/**/ PidVCont *c) {
    return c->f;
}
