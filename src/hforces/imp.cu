#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/ker.h"
#include "d/q.h"
#include "d/api.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "forces/type.h"
#include "forces/imp.h"
#include "k/read.h"
#include "k/common.h"

#include "utils/kl.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "hforces/imp.h"
#include "hforces/dev.map.h"
#include "hforces/dev.h"

namespace hforces {

static void get_start(const SFrag sfrag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)sfrag[i].n + 15) / 16);
}

void interactions(const SFrag26 ssfrag, const Frag26 ffrag, const Rnd26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(ssfrag.d, /**/ start.d);
    n = 2 * start.d[26];
        
    KL(dev::force, (k_cnf(n)), (start, ssfrag, ffrag, rrnd, /**/ ff));
}

} // hforces
