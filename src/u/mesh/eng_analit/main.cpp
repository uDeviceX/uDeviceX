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

struct Quant {
    int n;
    double *eng, *nx, *ny, *nz, *mean, *gauss;
};
struct QuantScalars {
    Scalars *eng, *nx, *ny, *nz, *mean, *gauss;
};
struct Spherical { double *r, *theta, *phi; };
struct Shape { double a0, a1, a2, D; };
static const Shape shape0 = {0.0518, 2.0026, -4.491, -1};

struct Param {
    int nv;
    Vectors *pos;
};

static void quant_scalars_ini(Quant *u, QuantScalars **pq) {
    int n;
    QuantScalars *q;
    EMALLOC(1, &q);
    n = u->n;
    scalars_double_ini(n, u->eng, /**/ &q->eng);
    scalars_double_ini(n, u->nx, /**/ &q->nx);
    scalars_double_ini(n, u->ny, /**/ &q->ny);
    scalars_double_ini(n, u->nz, /**/ &q->nz);
    scalars_double_ini(n, u->mean, /**/ &q->mean);
    scalars_double_ini(n, u->gauss, /**/ &q->gauss);
    *pq = q;
}

static void quant_scalars_fin(QuantScalars *q) {
    scalars_fin(q->eng);
    scalars_fin(q->nx); scalars_fin(q->ny); scalars_fin(q->nz);
    scalars_fin(q->mean); scalars_fin(q->gauss);
    EFREE(q);
}

static void quant_ini(int n, Quant **pq) {
    Quant *q;
    EMALLOC(1, &q);
    EMALLOC(n, &q->eng);
    EMALLOC(n, &q->nx); EMALLOC(n, &q->ny); EMALLOC(n, &q->nz);
    EMALLOC(n, &q->mean); EMALLOC(n, &q->gauss);
    q->n = n;
    *pq = q;
}
static void quant_fin(Quant *q) {
    EFREE(q->eng);
    EFREE(q->nx); EFREE(q->ny); EFREE(q->nz);
    EFREE(q->mean); EFREE(q->gauss);
    EFREE(q);
}

static void get(Vectors *pos, int i, /**/ double *px, double *py, double *pz) {
    enum {X, Y, Z};
    float r[3];
    vectors_get(pos, i, /**/ r);
    *px = r[X]; *py = r[Y]; *pz = r[Z];
}

static double sq(double x) { return x*x; }
static double sqrt0(double x) { return x > 0 ? sqrt(x) : 0; };

enum {EIG_OK, EIG_D};
static int eig(double a, double b, double c, /**/ double *px, double *py) { /* eigenvalues(matrix([a, b], [b, c])); */
    double D, x, y;
    D = c*c-2*a*c+4*b*b+a*a;
    if (D < 0) return EIG_D; 
    x = -(sqrt(D)-c-a)/2; /* TODO: not robust */
    y =  (sqrt(D)+c+a)/2;

    *px = x; *py = y;
    return EIG_OK;
}

static double f2(double r, const Shape *q) { /* diff(f, r, 2) */
    double a0, a1, a2;
    a0 = q->a0; a1 = q->a1; a2 = q->a2;
    return -(2*(6*r*(a2*(5*r-2)+a1)+a2-2*(a1+a0)))/(sqrt0(1-4*r)*(4*r-1));
}
static double f1(double r, const Shape *q) { /* diff(f, r) */
    double a0, a1, a2;
    a0 = q->a0; a1 = q->a1; a2 = q->a2;
    return sqrt(1-4*r)*(2*a2*r+a1)-(2*(r*(a2*r+a1)+a0))/sqrt0(1-4*r);
}
static double f(double r, const Shape *q) {
    double a0, a1, a2;
    a0 = q->a0; a1 = q->a1; a2 = q->a2;
    return sqrt0(1 - 4*r) * (a0 + a1*r + a2*r*r);
}
static double r(double x, double y, const Shape *q) {
    double D;
    D = q->D;
    return (x*x + y*y)/sq(D);
}
static double zrbc(double x, double y, const Shape *q) {
    double r0, D;
    D = q->D;
    r0 = r(x, y, q);
    return f(r0, q) * D;
}
static void normal(double x, double y, Shape *shape, /**/
                   double *pnx, double *pny, double *pnz) {
    enum {X, Y, Z};
    double D, f10, r0, nx, ny, nz;
    D = shape->D;
    r0 = r(x, y, shape);
    f10 = f1(r0, shape);
    nx = -(2*f10*x)/D;
    ny = -(2*f10*y)/D;
    nz = 1;

    *pnx = nx; *pny = ny; *pnz = nz;
}
static double norm0(const Shape *shape, int n, Vectors *pos) {
    int i;
    double x, y, z, z0, ans;
    ans = 0.0;
    for (i = 0; i < n; i++) {
        get(pos, i, /**/ &x, &y, &z);
        z0 = zrbc(x, y, shape);
        if (z < 0) z0 = -z0;
        ans += sq(z - z0);
    }
    return ans/n;
}
static void curv(double x, double y, Shape *shape, /**/ double *pL, double *pM, double *pN) {
    double L, M, N;
    double D, r0, f10, f20;
    double nx, ny, nz, n;

    D = shape->D;
    r0 = r(x, y, shape);
    f10 = f1(r0, shape);
    f20 = f2(r0, shape);

    L = (4*f20*pow(x,2))/pow(D,3)+(2*f10)/D;
    M = (4*f20*x*y)/pow(D,3);
    N = (4*f20*pow(y,2))/pow(D,3)+(2*f10)/D;

    normal(x, y, shape, &nx, &ny, &nz);
    n = sqrt(nx*nx + ny*ny + nz*nz);
    L /= n; M /= n; N /= n;

    *pL = L; *pM = M; *pN = N;
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
static double norm(const Shape *shape, Param *p) { return norm0(shape, p->nv, p->pos); }
static void fit(int nv, Vectors *pos, /**/ Shape *pshape, double *pno) {
    double mi, ma, D, no;
    Param param;
    Shape shape;
    param.nv = nv; param.pos = pos; shape = shape0;
    pos_min_max(nv, pos, /**/ &mi, &ma);

    D = ma - mi;
    shape.D = D;
    no = norm(&shape, &param);

    *pshape = shape; *pno = no;
}

static void compute_eng(int n, double *mean, /**/ double *eng) {
    int i;
    for (i = 0; i < n; i++)
        eng[i] = mean[i]*mean[i];
}

Shape fit_shape(int nv, Vectors *pos) {
    double err;
    Shape shape;
    fit(nv, pos, /**/ &shape, &err);
    msg_print("diameter, error: %g %g", shape.D, err);
    return shape;
}

static void compute_normal(Shape *shape, int n, Vectors *pos, /**/ double *nx, double *ny, double *nz) {
    /* normal is not normalized! */
    int i;
    double x, y, z;
    for (i = 0; i < n; i++) {
        get(pos, i, &x, &y, &z);
        normal(x, y, shape, &nx[i], &ny[i], &nz[i]);
    }
}

static void compute_curv(Shape *shape, int n, Vectors *pos, /**/ double *mean, double *gauss) {
    int i, status;
    double L, M, N, k0, k1;
    double x, y, z;
    for (i = 0; i < n; i++) {
        get(pos, i, &x, &y, &z);
        curv(x, y, shape, &L, &M, &N);
        status = eig(L, M, N, /**/ &k0, &k1);
        if (status != EIG_OK)
            ERR("eig fails for: %g %g %g", L, M, N);
        mean[i] = (k0 + k1)/2;
        gauss[i] = k0 * k1;
    }
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

static void dump_vtk(int nm, Quant *quant, Vectors *vectors, Out *out) {
    int id;
    VTKConf *conf;
    VTK *vtk;
    MeshRead *mesh;
    const char *path;
    MPI_Comm comm;
    QuantScalars *scalars;

    mesh = out->mesh; comm = out->comm; path = out->path;
    quant_scalars_ini(quant, /**/ &scalars);

    vtk_conf_ini(mesh, &conf);
    vtk_conf_vert(conf, "eng");
    vtk_conf_vert(conf, "nx");
    vtk_conf_vert(conf, "ny");
    vtk_conf_vert(conf, "nz");
    vtk_conf_vert(conf, "mean");
    vtk_conf_vert(conf, "gauss");

    vtk_ini(comm, nm, path, conf, &vtk);
    vtk_points(vtk, nm, vectors);
    vtk_vert(vtk, nm, scalars->eng, "eng");
    vtk_vert(vtk, nm, scalars->nx, "nx");
    vtk_vert(vtk, nm, scalars->ny, "ny");
    vtk_vert(vtk, nm, scalars->nz, "nz");
    vtk_vert(vtk, nm, scalars->mean, "mean");
    vtk_vert(vtk, nm, scalars->gauss, "gauss");    

    id = 0;
    vtk_write(vtk, comm, id);

    quant_scalars_fin(scalars);
    vtk_fin(vtk);
    vtk_conf_fin(conf);
}

static void main0(const char *cell, Out *out) {
    int nv, nm;
    const float *vert;
    Vectors  *pos;
    MeshSpherical *spherical;
    Spherical *sph;
    Quant *quant;
    Shape shape;
    nm = 1;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    nv = mesh_read_get_nv(out->mesh);
    quant_ini(nv, &quant);
    spherical_ini(nv, &sph);
    UC(mesh_spherical_ini(nv, /**/ &spherical));

    vert = mesh_read_get_vert(out->mesh);
    UC(vectors_float_ini(nv, vert, /**/ &pos));

     shape = fit_shape(nv, pos);

    compute_normal(&shape, nv, pos, /**/ quant->nx, quant->ny, quant->nz);
    compute_curv(&shape, nv, pos, /**/ quant->mean, quant->gauss);
    compute_eng(nv, quant->mean, /**/ quant->eng);

    mesh_spherical_apply(spherical, nm, pos, /**/ sph->r, sph->theta, sph->phi);
    dump_vtk(nm, quant, pos, out);
    dump_txt(nv, nm, sph, quant->eng);

    UC(quant_fin(quant));
    UC(vectors_fin(pos));
    UC(mesh_spherical_fin(spherical));
    UC(mesh_read_fin(out->mesh));
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
