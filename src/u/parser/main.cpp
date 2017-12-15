#include <stdio.h>

#include "msg.h"
#include "mpi/glb.h"
#include "utils/errors.h"
#include "parser/imp.h"

int main(int argc, char **argv) {
    Config *cfg;
    m::ini(&argc, &argv);

    conf_ini(/**/ &cfg);
    conf_read(argc, argv, /**/ cfg);

    conf_destroy(/**/ cfg);    

    m::fin();
}
