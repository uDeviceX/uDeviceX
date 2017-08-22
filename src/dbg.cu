#include <conf.h>
#include "inc/conf.h"
#include "inc/type.h"

#include "kl.h"

#include "dbg.h"
#include "dbg/dev.h"

namespace dbg {

void check_pp(const Particle *pp, int n) {
    KL(dev::check_pp, (k_cnf(n)), (pp, n));
}

} // dbg
