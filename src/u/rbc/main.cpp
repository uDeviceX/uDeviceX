#include <stdio.h>
#include <vector_types.h>

#include "inc/type.h"
#include "rbc/type.h"

#include "msg.h"
#include "mpi/glb.h"

void main0() {
    rbc::Quants q;
    MSG("rbc");
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0();
    m::fin();
}
