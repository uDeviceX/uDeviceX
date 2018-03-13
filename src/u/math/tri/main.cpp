#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "mpi/glb.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/tri/imp.h"

int main(int argc, char **argv) {
    int dims[3];
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims); /* eat args */
    m::fin();
}
