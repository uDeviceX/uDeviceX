#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/mc.h"
#include "utils/imp.h"

#include "io/off/imp.h"

#include "mpi/wrapper.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi rank/size: %d/%d", m::rank, m::size);
    m::fin();
}
