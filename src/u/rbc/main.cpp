#include <stdio.h>
#include <assert.h>

#include "mpi/glb.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}
