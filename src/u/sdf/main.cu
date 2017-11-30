#include <stdio.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/ker.h"
#include "d/api.h"
#include "msg.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "inc/macro.h"
#include "utils/kl.h"

#include "sdf/imp.h"
#include "sdf/type.h"
#include "sdf/imp/type.h"
#include "sdf/sub/dev/main.h"


namespace dev {
#include "dev.h"
}

void main0(sdf::Quants *qsdf) {
    float x, y, z;
    x = y = z = 0;
    KL(dev::main, (1, 1), (qsdf->texsdf, x, y, z));
}

void main1() {
    sdf::Quants *qsdf;
    sdf::alloc_quants(&qsdf);
    sdf::ini(m::cart, qsdf);
    main0(qsdf);
    sdf::free_quants(qsdf);
    dSync();
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main1();
    MSG("Hello world!");
    m::fin();
}
