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

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/mpi.h"
#include "inc/macro.h"

#include "kl.h"
#include "k/common.h"
#include "dev.h"

void pack_pp(const Particle *pp, int n, int **iidx, int *strt, float2 **dev) {
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, iidx, strt, /**/ dev));
}

