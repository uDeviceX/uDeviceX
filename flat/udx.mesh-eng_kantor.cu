// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/mesh/eng_kantor/imp.cpp */
#include <stdio.h>
#include <math.h>

/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

/* begin src/io/mesh_read/imp.h */
struct int4;
struct MeshRead;

// tag::interface[]
void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
// end::interface[]
/* end src/io/mesh_read/imp.h */
/* begin src/algo/scalars/imp.h */
struct Particle;
struct Scalars;
struct Vectors;

// tag::interface[]
void scalars_float_ini(int n, const float*, /**/ Scalars**);
void scalars_double_ini(int n, const double*, /**/ Scalars**);
void scalars_vectors_ini(int n, const Vectors*, int dim, /**/ Scalars**);

void scalars_zero_ini(int n, /**/ Scalars**);
void scalars_one_ini(int n, /**/ Scalars**);

void scalars_fin(Scalars*);
double scalars_get(const Scalars*, int i);
// end::interface[]
/* end src/algo/scalars/imp.h */
/* begin src/algo/vectors/imp.h */
struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
// end::interface[]
/* end src/algo/vectors/imp.h */

/* begin src/math/tri/imp.h */
namespace tri_hst {
// tag::interface[]
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
// end::interface[]
}
/* end src/math/tri/imp.h */
/* begin src/mesh/angle/imp.h */
struct MeshAngle;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);
void  mesh_angle_apply(MeshAngle*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/angle/imp.h */

/* begin src/mesh/eng_kantor/imp.h */
struct MeshEngKantor;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_eng_kantor_ini(MeshRead*, int max_nm, /**/ MeshEngKantor**); // <1>
void mesh_eng_kantor_fin(MeshEngKantor*);

void  mesh_eng_kantor_apply(MeshEngKantor*, int nm, Vectors*, double kb, double ang, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/eng_kantor/imp.h */
/* begin src/mesh/eng_kantor/imp/type.h */
struct MeshEngKantor {
    int max_nm; /* maximum number of meshes */
    int nv, ne;
    double *angles;
    MeshAngle  *angle;
};
/* end src/mesh/eng_kantor/imp/type.h */
/* begin src/mesh/eng_kantor/imp/main.h */
void mesh_eng_kantor_ini(MeshRead *mesh, int nm, /**/ MeshEngKantor **pq) {
    int nv, ne;
    MeshEngKantor *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->ne = ne;
    EMALLOC(nm*ne, &q->angles);
    mesh_angle_ini(mesh, /**/ &q->angle);

    *pq = q;
}

void mesh_eng_kantor_fin(MeshEngKantor *q) {
    EFREE(q->angles);
    mesh_angle_fin(q->angle);
    EFREE(q);
}

void mesh_eng_kantor_apply(MeshEngKantor *q, int nm, Vectors *pos, double kb, double angle0, /**/ double *o) {
    int i, ne;
    double *angles, angle;
    angles = q->angles;
    ne = q->ne;
    mesh_angle_apply(q->angle, nm, pos, /**/ angles);
    for (i = 0; i < nm * ne; i++) {
        angle = angles[i];
        o[i] = 2*kb*(1 - cos(angle - angle0));
    }
}
/* end src/mesh/eng_kantor/imp/main.h */
/* end src/mesh/eng_kantor/imp.cpp */

#line 1 "src/mesh/force/kantor0/imp.cpp"
