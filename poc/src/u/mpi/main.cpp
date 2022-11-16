#include <stdio.h>
#include <conf.h>
#include <mpi.h>
#include "inc/conf.h"

#include "utils/msg.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    MC(m::Barrier(-1234)); /* trigger an error */
    m::fin();
}
