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
#include "mesh/cylindrical/imp.h"

#include "io/vtk/imp.h"
#include "algo/scalars/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

struct Cylindrical {
    double *r, *phi, *z;
};

void cylindrical_ini(int n, Cylindrical **pq) {
    Cylindrical *q;
    EMALLOC(1, &q);
    EMALLOC(n, &q->r); EMALLOC(n, &q->phi); EMALLOC(n, &q->z);
    *pq = q;
}

void cylindrical_fin(Cylindrical *q) {
    EFREE(q->r); EFREE(q->phi); EFREE(q->z);
    EFREE(q);
}

static void dump_txt(int nv, int nm, Cylindrical *cyl, const double *a) {
    int n, i;
    n = nv * nm;
    for (i = 0; i < n; i++)
        printf("%g %g %g %g\n", cyl->r[i], cyl->phi[i], cyl->z[i], a[i]);
}

static void dump_vtk(int nv, int nm, const double *eng, Vectors *vectors, Out *out) {
    int id;
    VTKConf *conf;
    VTK *vtk;
    MeshRead *mesh;
    const char *path;
    MPI_Comm comm;
    Scalars *scalars;

    mesh = out->mesh; comm = out->comm; path = out->path;
    scalars_double_ini(nv, eng, /**/ &scalars);

    vtk_conf_ini(mesh, &conf);
    vtk_conf_vert(conf, "eng");

    vtk_ini(comm, nm, path, conf, &vtk);
    vtk_points(vtk, nm, vectors);
    vtk_vert(vtk, nm, scalars, "eng");
    id = 0;
    vtk_write(vtk, comm, id);

    scalars_fin(scalars);
    vtk_fin(vtk);
    vtk_conf_fin(conf);
}

static void main0(const char *cell, Out *out) {
    int nv, nm;
    MeshEngJulicher *eng_julicher;
    MeshCylindrical *cylindrical;
    const float *vert;
    Vectors  *pos;
    double *eng, kb;
    Cylindrical *cyl;

    nm = 1; kb = 1;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_eng_julicher_ini(out->mesh, nm, /**/ &eng_julicher));
    nv = mesh_read_get_nv(out->mesh);
    cylindrical_ini(nv, &cyl);
    UC(mesh_cylindrical_ini(nv, /**/ &cylindrical));
    vert = mesh_read_get_vert(out->mesh);
    UC(vectors_float_ini(nv, vert, /**/ &pos));

    EMALLOC(nv, &eng);
    mesh_eng_julicher_apply(eng_julicher, nm, pos, kb, /**/ eng);
    mesh_cylindrical_apply(cylindrical, nm, pos, /**/ cyl->r, cyl->phi, cyl->z);
    dump_vtk(nv, nm, eng, pos, out);
    dump_txt(nv, nm, cyl, eng);

    mesh_eng_julicher_fin(eng_julicher);
    cylindrical_fin(cyl);
    UC(mesh_cylindrical_fin(cylindrical));
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
