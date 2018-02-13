#include <mpi.h>
#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "utils/kl.h"

namespace dev {
#include "dev.h"
}

void main0() {
    KL(dev::main, (1, 1), ());
    dSync();
}

int main(int argc, char **argv) {
    int rank, size;
    m::ini(&argc, &argv);
    MC(m::Comm_rank(m::cart, &rank));
    MC(m::Comm_size(m::cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);
    main0();
    msg_print("Hello world!");
    m::fin();
}
