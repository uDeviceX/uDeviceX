#include <stdio.h>
#include <conf.h>
#include <mpi.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"
#include "glb/imp.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MC(m::Barrier(-1234));
    m::fin();
}
