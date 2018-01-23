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

#include "d/api.h"
#include "mpi/wrapper.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "utils/mc.h"
#include "utils/cc.h"

#include "coords/imp.h"

#include "partlist/type.h"
#include "algo/scan/imp.h"
#include "clist/imp.h"

#include "io/restart/imp.h"
#include "inter/color.h"

#include "io/punto/imp.h"

#include "imp.h"

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/generate.h"
#include "imp/start.h"
#include "imp/punto.h"

/* TODO does it belong here? */
void flu_build_cells(/**/ FluQuants *q) {
    clist_build(q->n, q->n, q->pp, /**/ q->pp0, &q->cells, q->mcells);
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}
