#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/conf.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "kl.h"

#include "dbg.h"
#include "dbg/dev.h"

namespace dbg {

void check_pp(const Particle *pp, int n) {
    KL(dev::check_pp, (k_cnf(n)), (pp, n));
}

} // dbg
