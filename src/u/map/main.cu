#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"

#include "utils/kl.h"

namespace dev {
#include "dev.h"
}

void main0() { KL(dev::main, (1, 1), ()); }

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    MSG("Hello world!");
    m::fin();
}
