#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "utils/cc.h"

#include "mpi/glb.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "parser/imp.h"

#include "io/txt/imp.h"

static void read_pp(const char *fname) {
    TxtRead *tr;
    UC(txt_read_pp(fname, &tr));
    UC(txt_read_fin(tr));
}

int main(int argc, char **argv) {
    Config *cfg;
    const char *fname;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(conf_lookup_string(cfg, "fname", &fname));
    UC(read_pp(fname));
    
    
    UC(conf_fin(cfg));
    m::fin();
}
