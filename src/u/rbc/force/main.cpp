#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi/glb.h"
#include "parser/imp.h"
#include "rbc/params/imp.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    Config *cfg;
    RbcParams *par;
    const char *cell, *ic;
    m::ini(&argc, &argv);
    conf_ini(&cfg);
    conf_read(argc, argv, cfg);

    conf_lookup_string(cfg, "rbc.cell", &cell);
    conf_lookup_string(cfg, "rbc.ic", &ic);

    rbc_params_ini(&par);
    rbc_params_set_conf(cfg, par);
    
    run(cell, ic, par);

    rbc_params_fin(par);
    conf_fin(cfg);
    m::fin();
}
