// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/grid/xmf/imp.cpp */
#include <string.h>
#include <stdio.h>
#include <vector_types.h>

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

/* begin src/io/grid/xmf/imp.h */
// tag::int[]
void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names);
// end::int[]
/* end src/io/grid/xmf/imp.h */

static void header(FILE *f) {
    fprintf(f, "<?xml version=\"1.0\" ?>\n");
    fprintf(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(f, "<Xdmf Version=\"2.0\">\n");
    fprintf(f, " <Domain>\n");
}

static void epilogue(FILE *f) {
    fprintf(f, " </Domain>\n");
    fprintf(f, "</Xdmf>\n");
}

static float3 get_spacing(int3 D, int3 G) {
    float3 d;
    d.x = (float) D.x / (float) G.x;
    d.y = (float) D.y / (float) G.y;
    d.z = (float) D.z / (float) G.z;
    return d;
}

static void grid(int3 D, int3 G, FILE *f, const char *path, int n, const char **names) {
    int i;
    int base_len = strlen(path) - strlen(".raw");
    float3 o, d; /* origin, spacing */
    o.x = o.y = o.z = 0.0;
    d = get_spacing(D, G);

    fprintf(f, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(f, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G.z, 1 + G.y, 1 + G.x);
    fprintf(f, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(f, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", o.x, o.y, o.z);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", d.x, d.y, d.z);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "     </Geometry>\n");
    for(i = 0; i < n; ++i) {
        fprintf(f, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", names[i]);
        fprintf(f, "       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"Binary\">\n", G.z, G.y, G.x);
        fprintf(f, "        %.*s.%s.raw\n", base_len, path, names[i]);
        fprintf(f, "       </DataItem>\n");
        fprintf(f, "     </Attribute>\n");
    }
    fprintf(f, "   </Grid>\n");
}

static int eq(const char *a, const char *b) { return !strcmp(a, b); }
static void xsuffix(const char *i, /**/ char *o) {
    /* replace suffix .raw by .xmf */
    char s[] = ".raw";
    int n;
    n = strlen(i) - strlen(s);
    if (n >= 0 && eq(i + n, s)) {strncpy(o, i, n); o[n] = '\0';}
    else                        strcpy (o, i);
    strcat(o, ".xmf");
}
static void basename(const char *i, /**/ char *o) {
    const char *p;
    p = i;
    while (*i != '\0') if (*i++ == '/') p = i;
    strcpy(o, p);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names) {
    char w[BUFSIZ];
    FILE *f;

    xsuffix(path, /**/ w);
    UC(efopen(w, "w", /**/ &f));
    header(f);

    basename(path, /**/ w);
    grid(domainSize, gridSize, f, w, ncomp, names);
    epilogue(f);
    UC(efclose(f));
}
/* end src/io/grid/xmf/imp.cpp */

#line 1 "src/io/mesh/imp.cpp"
