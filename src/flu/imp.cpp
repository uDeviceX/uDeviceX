#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/os.h"
#include "utils/msg.h"
#include "inc/def.h"
#include "inc/dev.h"
#include "inc/type.h"

#include "glob/type.h"

#include "d/api.h"
#include "mpi/wrapper.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "utils/mc.h"
#include "utils/cc.h"

#include "partlist/type.h"
#include "algo/scan/imp.h"
#include "clist/imp.h"

#include "io/restart/imp.h"
#include "inter/color.h"

#include "imp.h"

namespace flu {

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/generate.h"
#include "imp/start.h"

/* TODO does it belong here? */
void build_cells(/**/ Quants *q) {
    clist::build(q->n, q->n, q->pp, /**/ q->pp0, &q->cells, &q->mcells);
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}

} // flu
