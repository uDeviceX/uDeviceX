#include <stdio.h>
#include <assert.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "msg.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "utils/te.h"
#include "utils/texo.h"

#include "rbc/type.h"
#include "rbc/main/imp.h"
#include "rbc/force/imp.h"

#include "mpi/glb.h"

#include "imp.h"

static void run0(const char *cell, const char *ic, rbc::Quants q) {
    rbc::force::TicketT tt;
    rbc::main::gen_quants(cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &tt);
    rbc::force::fin_ticket(&tt);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run0(cell, ic, q);
    rbc::main::fin(&q);
}
