#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "algo/vectors/imp.h"
#include "mesh/eng_julicher/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

static void dump(int nv, int nm, double *data, Vectors*, Out*) {
    int i, n;
    n = nv * nm;
    for (i = 0; i < n; i++)
        printf("%g\n", data[i]);
}

static void main0(const char *cell, Out *out) {
    int nv, nm;
    MeshEngJulicher *eng_julicher;
    Vectors  *pos;
    double *eng;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_eng_julicher_ini(out->mesh, &eng_julicher));
    nv = mesh_read_get_nv(out->mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(out->mesh), /**/ &pos));

    nm = 1;
    EMALLOC(nv, &eng);
    mesh_eng_julicher_apply(eng_julicher, nm, pos, /**/ eng);
    dump(nv, nm, eng, pos, out);

    mesh_eng_julicher_fin(eng_julicher);
    UC(vectors_fin(pos));
    UC(mesh_read_fin(out->mesh));
    EFREE(eng);
}

int main(int argc, char **argv) {
    const char *i;
    Out out;
    Config *cfg;
    int rank, size, dims[3];
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &out.comm);

    MC(m::Comm_rank(out.comm, &rank));
    MC(m::Comm_size(out.comm, &size));

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "i", &i));
    UC(conf_lookup_string(cfg, "o", &out.path));

    main0(i, &out);

    UC(conf_fin(cfg));
    MC(m::Barrier(out.comm));
    m::fin();
}
