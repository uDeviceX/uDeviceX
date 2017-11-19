#include <stdio.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/basetags.h"
#include "frag/imp.h"

#include "comm/oc/imp.h"
#include "comm/imp.h"

void main0() {
    OC(comm::comm_error());
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    MSG("Hello world!");
    m::fin();
}
