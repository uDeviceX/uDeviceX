#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/conf.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "cc.h"
#include "msg.h"
#include "d/api.h"

#include "kl.h"

#include "dbg.h"
#include "dbg/error.h"
#include "dbg/dev.h"

#include "dbg/switch.h"

namespace dbg {

namespace sub {

void hard_check_pp(const Particle *pp, int n) {
    KL(dev::hard_check_pp, (k_cnf(n)), (pp, n));
}

void hard_check_ff(const Force *ff, int n) {
    KL(dev::hard_check_ff, (k_cnf(n)), (ff, n));
}

} // sub

void check_pp(const Particle *pp, int n, const char *M) {
    DBG(sub::hard_check_pp, (pp, n), M);
}
void check_ff(const Force *ff, int n, const char *M) {
    DBG(sub::hard_check_ff, (ff, n), M);
}

} // dbg
