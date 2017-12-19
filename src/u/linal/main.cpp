#include <stdio.h>

#include "mpi/glb.h"
#include "msg.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    MSG("Hello world!");
    m::fin();
}
