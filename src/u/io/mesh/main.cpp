#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "utils/error.h"

#include "io/off/imp.h"
#include "io/mesh/imp.h"

#include "parser/imp.h"

static void write(const char *o, OffRead *cell) {
    int nc;
    MeshWrite *mesh;
    Coords *c;
    MPI_Comm cart;
    Particle *pp;

    cart = m::cart;
    UC(mesh_write_ini_off(cell, o, /**/ &mesh));
    UC(coords_ini(cart, XS, YS, ZS, /**/ &c));

    nc = 1; pp = NULL;
    UC(mesh_write_dump(mesh, cart, c, nc, pp, 0));

    UC(coords_fin(c));
    UC(mesh_write_fin(mesh));
}

static void log(OffRead *cell) {
    int nv, nt, md;
    md = off_get_md(cell);
    nv = off_get_nv(cell);
    nt = off_get_nt(cell);
    msg_print("nv, nt, max degree: %d %d %d", nv, nt, md);
}

static void main0(Config *c) {
    OffRead *cell;
    const char *i, *o; /* input and output */
    UC(conf_lookup_string(c, "i", &i));
    UC(conf_lookup_string(c, "o", &o));

    msg_print("i = '%s'", i);
    msg_print("o = '%s'", o);
    UC(off_read(i, &cell));
    UC(log(cell));
    write(o, cell);
    UC(off_fin(cell));
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
