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

static void main0(rbc::Quants q) {
    rbc::force::TicketT tt;
    rbc::main::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &q);
    rbc::force::gen_ticket(q, &tt);
    rbc::force::fin_ticket(&tt);
}

void main1() {
    rbc::Quants q;
    rbc::main::ini(&q);
    main0(q);
    rbc::main::fin(&q);
}
