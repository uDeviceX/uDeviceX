#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "inc/dev.h"
#include "inc/type.h"

#include "d/api.h"
#include "mpi/wrapper.h"

#include "utils/mc.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "msg.h"

#include "algo/scan/int.h"
#include "clist/imp.h"
#include "rnd/imp.h"

#include "io/restart.h"
#include "inter/color.h"

#include "imp.h"

namespace flu {

#include "dev.h"

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/generate.h"
#include "imp/start.h"

/* TODO does it belong here? */
void build_cells(/**/ Quants *q) {
    clist::build(q->n, q->n, q->pp, /**/ q->pp0, &q->cells, &q->tcells);
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}

} // flu
