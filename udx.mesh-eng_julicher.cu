// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/mesh/eng_julicher/imp.cpp */
#include <stdio.h>

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

/* begin src/mesh/edg_len/imp.h */
struct MeshEdgLen;
struct MeshRead;
struct Vectors;

// tag::interface[]
void mesh_edg_len_ini(MeshRead*, MeshEdgLen**); // <1>
void mesh_edg_len_fin(MeshEdgLen*);
void mesh_edg_len_apply(MeshEdgLen*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/edg_len/imp.h */
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
/* begin src/mesh/vert_area/imp.h */
struct MeshVertArea;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_vert_area_ini(MeshRead*, /**/ MeshVertArea**); // <1>
void mesh_vert_area_fin(MeshVertArea*);

void  mesh_vert_area_apply(MeshVertArea*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/vert_area/imp.h */
/* begin src/mesh/scatter/imp.h */
struct MeshScatter;
struct MeshRead;
struct Scalars;

// tag::interface[]
void mesh_scatter_ini(MeshRead*, MeshScatter**); // <1>
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
void mesh_scatter_edg2vert_avg(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/scatter/imp.h */

/* begin src/mesh/eng_julicher/imp.h */
struct MeshEngJulicher;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_eng_julicher_ini(MeshRead*, int max_nm, /**/ MeshEngJulicher**); // <1>
void mesh_eng_julicher_fin(MeshEngJulicher*);

void  mesh_eng_julicher_apply(MeshEngJulicher*, int nm, Vectors*, double kb, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/eng_julicher/imp.h */
/* begin src/mesh/eng_julicher/imp/type.h */
struct MeshEngJulicher {
    int max_nm; /* maximum number of meshes */
    int nv, ne;

    /* edge lens, edge angles, vert. areas, vert. curvatures */
    double *lens, *angles, *areas, *curvs_edg, *curvs_vert;

    MeshEdgLen *len;
    MeshAngle  *angle;
    MeshVertArea *area;

    MeshScatter *scatter;
};
/* end src/mesh/eng_julicher/imp/type.h */
/* begin src/mesh/eng_julicher/imp/main.h */
void mesh_eng_julicher_ini(MeshRead *mesh, int nm, /**/ MeshEngJulicher **pq) {
    int nv, ne;
    MeshEngJulicher *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->ne = ne;
    EMALLOC(nm*ne, &q->lens);
    EMALLOC(nm*ne, &q->angles);
    EMALLOC(nm*nv, &q->areas);
    EMALLOC(nm*ne, &q->curvs_edg);
    EMALLOC(nm*nv, &q->curvs_vert);

    mesh_edg_len_ini(mesh, /**/ &q->len);
    mesh_angle_ini(mesh, /**/ &q->angle);
    mesh_vert_area_ini(mesh, /**/ &q->area);
    mesh_scatter_ini(mesh, /**/ &q->scatter);

    *pq = q;
}

void mesh_eng_julicher_fin(MeshEngJulicher *q) {
    EFREE(q->lens);
    EFREE(q->angles);
    EFREE(q->areas);
    EFREE(q->curvs_edg);
    EFREE(q->curvs_vert);

    mesh_edg_len_fin(q->len);
    mesh_angle_fin(q->angle);
    mesh_vert_area_fin(q->area);
    mesh_scatter_fin(q->scatter);

    EFREE(q);
}

void mesh_eng_julicher_apply(MeshEngJulicher *q, int nm, Vectors *pos, double kb, /**/ double *o) {
    int i, nv, ne;
    double *lens, *angles, *areas, *curvs_vert, *curvs_edg;
    double area, curv;
    Scalars *sc;
    lens = q->lens; angles = q->angles; areas = q->areas;
    curvs_edg = q->curvs_edg; curvs_vert = q->curvs_vert;

    ne = q->ne; nv = q->nv;
    mesh_edg_len_apply(q->len, nm, pos, /**/ lens);
    mesh_angle_apply(q->angle, nm, pos, /**/ angles);
    for (i = 0; i < nm * ne; i++)
        curvs_edg[i] = lens[i]*angles[i];

    scalars_double_ini(nm * ne, curvs_edg, /**/ &sc);
    mesh_scatter_edg2vert(q->scatter, nm, sc, /**/ curvs_vert);
    for (i = 0; i < nm * nv; i++)
        curvs_vert[i] /= 4;

    mesh_vert_area_apply(q->area, nm, pos, /**/ areas);
    for (i = 0; i < nm * nv; i++) {
        curv = curvs_vert[i];
        area = areas[i];
        if (area <= 0) ERR("area[%d]=%g <= 0", area, i);
        o[i] = 2*kb*curv*curv/area;
    }
    scalars_fin(sc);
    mesh_vert_area_apply(q->area, nm, pos, /**/ areas);
}
/* end src/mesh/eng_julicher/imp/main.h */
/* end src/mesh/eng_julicher/imp.cpp */

#line 1 "src/mesh/eng_kantor/imp.cpp"
