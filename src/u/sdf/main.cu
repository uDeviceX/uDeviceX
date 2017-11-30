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

#include "sdf/type.h"
#include "sdf/int.h"

#include "utils/kl.h"

namespace dev {
#include "dev.h"
}

void main0() {
    sdf::Quants qsdf;
    sdf::alloc_quants(&qsdf);
    sdf::ini(m::cart, &qsdf);

    KL(dev::main, (1, 1), ());
    dSync();

    sdf::free_quants(&qsdf);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    MSG("Hello world!");
    m::fin();
}
