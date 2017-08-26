#include <assert.h>
#include <mpi.h>
#include "l/m.h"
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "common.h"
#include "msg.h"
#include "m.h"
#include "cc.h"

#include "dual/type.h"
#include "dual/int.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/mpi.h"
#include "inc/macro.h"

#include "scan/int.h"
#include "odstr/halo/int.h"

#include "kl.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"

#include "frag.h"

#include "odstr/type.h"
#include "odstr/imp.h"

#include "odstr/mpi.h"
#include "odstr/mpi.ii.h"

#include "dev.h"

void pack_pp(const Particle *pp, int n, Send *s) {
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, s->iidx, s->strt, /**/ s->pp.dev));
}

void pack_ii(const int *ii, int n, const Send *s, Pbufs<int>* sii) {
    KL((dev::pack<int, 1>), (k_cnf(n)),(ii, s->iidx, s->strt, /**/ sii->dev));
}
