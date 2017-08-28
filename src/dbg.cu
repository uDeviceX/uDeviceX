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

#include "dbg/macro/switch.h"

namespace dbg {

namespace sub {

void check_pos(const Particle *pp, int n) {
    KL(dev::check_pos, (k_cnf(n)), (pp, n));
}

void check_vv(const Particle *pp, int n) {
    KL(dev::check_vv, (k_cnf(n)), (pp, n));
}

void check_pos_pu(const Particle *pp, int n) {
    KL(dev::check_pos_pu, (k_cnf(n)), (pp, n));
}

void check_ff(const Force *ff, int n) {
    KL(dev::check_ff, (k_cnf(n)), (ff, n));
}
} // sub

void check_pos(const Particle *pp, int n, const char *M) {
    DBG(sub::check_pos, (pp, n), M);
}
void check_vv(const Particle *pp, int n, const char *M) {
    DBG(sub::check_vv, (pp, n), M);
}
void check_pos_pu(const Particle *pp, int n, const char *M) {
    DBG(sub::check_pos_pu, (pp, n), M);
}
void check_ff(const Force *ff, int n, const char *M) {
    DBG(sub::check_ff, (ff, n), M);
}
} // dbg
