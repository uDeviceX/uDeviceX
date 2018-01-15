#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi/glb.h"
#include "parser/imp.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    Config *cfg;
    const char *cell, *ic;
    m::ini(&argc, &argv);
    conf_ini(&cfg);
    conf_read(argc, argv, cfg);

    conf_lookup_string(cfg, "rbc.cell", &cell);
    conf_lookup_string(cfg, "rbc.ic", &ic);

    run(cell, ic);
    
    conf_fin(cfg);
    m::fin();
}
