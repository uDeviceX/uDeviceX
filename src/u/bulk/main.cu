#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"

#include "mpi/glb.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "utils/kl.h"

namespace dev {
#include "dev.h"
}

void main0() {
    KL(dev::main, (1, 1), ());
    dSync();
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);
    main0();
    msg_print("Hello world!");
    m::fin();
}
