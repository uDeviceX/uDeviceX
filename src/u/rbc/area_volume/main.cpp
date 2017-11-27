#include <stdio.h>
#include <string.h>

#include "mpi/glb.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    run("rbc.off", "rbcs-ic.txt");
    m::fin();
}
