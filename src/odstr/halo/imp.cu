#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "k/common.h"

#include "common.h"
#include "msg.h"
#include "m.h"
#include "cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/macro.h"

#include "kl.h"

#include "dev.h"

void halo(const Particle *pp, int n, /**/ int **iidx, int *sizes) {
    Dzero(sizes, 27);
    KL(dev::halo, (k_cnf(n)),(pp, n, /**/ iidx, sizes));
}
