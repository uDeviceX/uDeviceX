#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h" /* mini-MPI and -device */
#include "d/api.h"

#include "glb.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "utils/cc.h"


#include "algo/scan/int.h"
#include "clistx/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    m::fin();
}
