#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "msg.h"

#include "mpi/glb.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "utils/kl.h"
#include "utils/map/dev.h"

namespace dev {
#include "dev.h"
}

void main0() {
    KL(dev::main, (1, 1), ());
    dSync();
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    MSG("Hello world!");
    m::fin();
}
