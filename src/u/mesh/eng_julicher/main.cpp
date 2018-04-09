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
#include "mesh/spherical/imp.h"

#include "io/vtk/imp.h"
#include "algo/scalars/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

struct Spherical {
    double *r, *theta, *phi;
};

void spherical_ini(int n, Spherical **pq) {
    Spherical *q;
    EMALLOC(1, &q);
    EMALLOC(n, &q->r); EMALLOC(n, &q->theta); EMALLOC(n, &q->phi);
    *pq = q;
}

void spherical_fin(Spherical *q) {
    EFREE(q->r); EFREE(q->theta); EFREE(q->phi);
    EFREE(q);
}

static void dump_txt(int nv, int nm, Spherical *sph, const double *a) {
    int n, i;
    n = nv * nm;
    for (i = 0; i < n; i++)
        printf("%g %g %g %g\n", sph->r[i], sph->theta[i], sph->phi[i], a[i]);
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
    MeshSpherical *spherical;
    const float *vert;
    Vectors  *pos;
    double *eng, kb;
    Spherical *sph;

    nm = 1; kb = 1;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_eng_julicher_ini(out->mesh, nm, /**/ &eng_julicher));
    nv = mesh_read_get_nv(out->mesh);
    spherical_ini(nv, &sph);
    UC(mesh_spherical_ini(nv, /**/ &spherical));
    vert = mesh_read_get_vert(out->mesh);
    UC(vectors_float_ini(nv, vert, /**/ &pos));

    EMALLOC(nv, &eng);
    mesh_eng_julicher_apply(eng_julicher, nm, pos, kb, /**/ eng);
    mesh_spherical_apply(spherical, nm, pos, /**/ sph->r, sph->theta, sph->phi);
    dump_vtk(nv, nm, eng, pos, out);
    dump_txt(nv, nm, sph, eng);

    mesh_eng_julicher_fin(eng_julicher);
    spherical_fin(sph);
    UC(mesh_spherical_fin(spherical));
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
