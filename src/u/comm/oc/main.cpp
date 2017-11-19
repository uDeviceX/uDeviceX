#include <stdio.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/basetags.h"
#include "frag/imp.h"

#include "comm/oc/imp.h"
#include "comm/imp.h"

void main0() {
    using namespace comm;
    basetags::TagGen tg;
    Stamp stamp;
    int capacity[NBAGS];
    float maxdensity = 26;
    frag_estimates(NBAGS, maxdensity, /**/ capacity);
    ini(/**/ &tg);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    MSG("Hello world!");
    m::fin();
}
