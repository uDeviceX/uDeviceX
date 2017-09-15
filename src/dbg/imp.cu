#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/def.h"
#include "inc/conf.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "utils/cc.h"
#include "msg.h"
#include "d/api.h"

#include "utils/kl.h"

#include "dbg/imp.h"
#include "dbg/error.h"

namespace dbg {
namespace dev {
#include "dbg/dev/common.h"
#include "dbg/dev/pos.h"
#include "dbg/dev/vel.h"
#include "dbg/dev/force.h"
#include "dbg/dev/color.h"
#include "dbg/dev/clist.h"
} // dev
} // dbg

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

void check_cc(const int *cc, int n) {
    KL(dev::check_cc, (k_cnf(n)), (cc, n));
}
} // sub

void check_pos(const Particle *pp, int n, const char *file, int line, const char *M) {
    DBG(sub::check_pos, (pp, n), file, line, M);
}
void check_vv(const Particle *pp, int n, const char *file, int line, const char *M) {
    DBG(sub::check_vv, (pp, n), file, line, M);
}
void check_pos_pu(const Particle *pp, int n, const char *file, int line, const char *M) {
    DBG(sub::check_pos_pu, (pp, n), file, line, M);
}
void check_ff(const Force *ff, int n, const char *file, int line, const char *M) {
    DBG(sub::check_ff, (ff, n), file, line, M);
}
void check_cc(const int *cc, int n, const char *file, int line, const char *M) {
    DBG(sub::check_cc, (cc, n), file, line, M);
}
} // dbg
