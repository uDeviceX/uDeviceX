#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"

#include "io/off/imp.h"
#include "io/mesh/imp.h"

#include "parser/imp.h"

static void write(const char *o, OffRead *cell) {
    MeshWrite *m;
    UC(mesh_write_ini_off(cell, o, &m));
    UC(mesh_write_fin(m));
}

static void main0(Config *c) {
    int nv, nt, md;
    OffRead *cell;
    const char *i, *o; /* input and output */
    UC(conf_lookup_string(c, "i", &i));
    UC(conf_lookup_string(c, "o", &o));

    msg_print("i = '%s'", i);
    msg_print("o = '%s'", o);
    UC(off_read(i, &cell));

    write(o, cell)

    md = cell_get_md(cell);
    nv = cell_get_nv(cell);
    nt = cell_get_nt(cell);
    write(cell);
    
    msg_print("nv, nt, max degree: %d %d %d", nv, nt, md);
    UC(cell_fin(cell));
}

int main(int argc, char **argv) {
    Config *cfg;
    m::ini(&argc, &argv);
    msg_ini(m::rank);

    UC(conf_ini(/**/ &cfg));
    UC(conf_read(argc, argv, /**/ cfg));
    UC(main0(cfg));
    UC(conf_fin(cfg));
    m::fin();
}
