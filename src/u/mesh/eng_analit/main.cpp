#include <math.h>
#include <stdio.h>
#include <float.h>

#include <mpi.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "mesh/spherical/imp.h"

#include "io/vtk/imp.h"

#include "algo/vectors/imp.h"
#include "algo/scalars/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

struct Spherical {
    double *r, *theta, *phi;
};

struct Shape {
    double a0, a1, a2, D;
};

struct Param {
    int nv;
    Vectors *pos;
};

static void get(Vectors *pos, int i, /**/ double *px, double *py, double *pz) {
    enum {X, Y, Z};
    float r[3];
    vectors_get(pos, i, /**/ r);
    *px = r[X]; *py = r[Y]; *pz = r[Z];
}

static double sq(double x) { return x*x; }
static double sqrt0(double x) { return x > 0 ? sqrt(x) : 0; };
static double zrbc(double x, double y, Shape *q) {
    double a0, a1, a2, D, r, z;
    a0 = q->a0; a1 = q->a1; a2 = q->a2; D = q->D;
    r = x*x + y*y; r /= sq(D);
    z  = sqrt0(1 - 4*r) * (a0 + a1*r + a2*r*r);
    return z * D;
}
static double norm0(double D, int n, Vectors *pos) {
    int i;
    double x, y, z, z0, ans;
    Shape shape;
    shape.a0 = 0.0518; shape.a1 = 2.0026; shape.a2 = -4.491; shape.D =  D;
    ans = 0.0;
    for (i = 0; i < n; i++) {
        get(pos, i, /**/ &x, &y, &z);
        z0 = zrbc(x, y, &shape);
        if (z < 0) z0 = -z0;
        ans += sq(z - z0);
    }
    return ans/n;
}
static void pos_min_max(int n, Vectors *pos, double *pmi, double *pma) {
    double mi, ma, x, y, z;
    int i;
    mi = DBL_MAX; ma = -DBL_MAX;
    for (i = 0; i < n; i++) {
        get(pos, i, &x, &y, &z);
        if (x > ma) ma = x;
        if (x < mi) mi = x;
    }
    *pma = ma; *pmi = mi;
}
static double norm(double D, Param *p) { return norm0(D, p->nv, p->pos); }
static void fit(int nv, Vectors *pos, /**/ double *pD, double *pno) {
    double mi, ma, D, no;
    Param param;
    param.nv = nv; param.pos = pos;
    pos_min_max(nv, pos, /**/ &mi, &ma);
    
    D = ma - mi;
    no = norm(D, &param);
    *pD = D; *pno = no;
}

static void compute_eng(int nv, Vectors *pos, /**/ double *eng) {
    double D, err;
    int i;
    fit(nv, pos, /**/ &D, &err);
    msg_print("diameter, error: %g %g", D, err);
    for (i = 0; i < nv; i++) eng[i] = i;
}

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
    const float *vert;
    Vectors  *pos;
    MeshSpherical *spherical;
    Spherical *sph;
    double *eng;
    nm = 1;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    nv = mesh_read_get_nv(out->mesh);
    spherical_ini(nv, &sph);
    UC(mesh_spherical_ini(nv, /**/ &spherical));

    vert = mesh_read_get_vert(out->mesh);
    UC(vectors_float_ini(nv, vert, /**/ &pos));

    EMALLOC(nv, &eng);
    compute_eng(nv, pos, /**/ eng);

    mesh_spherical_apply(spherical, nm, pos, /**/ sph->r, sph->theta, sph->phi);

    dump_vtk(nv, nm, eng, pos, out);
    dump_txt(nv, nm, sph, eng);

    UC(vectors_fin(pos));
    UC(mesh_spherical_fin(spherical));
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
