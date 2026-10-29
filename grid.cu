#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>
struct Coords;
void coords_ini(MPI_Comm, int, int, int, Coords **);
void coords_fin(Coords *);
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int xdomain(const Coords *);
int ydomain(const Coords *);
int zdomain(const Coords *);
void error_stack_push(const char *, int);
void error_stack_pop();
void error_signal(const char *, int, const char *, ...);
bool error_get();
void error_report();
void error_abort();
void efopen(const char *, const char *, FILE **);
void efclose(FILE *);
namespace mpicheck { void check(int, const char *, int); }
namespace m {
int Cart_get(MPI_Comm, int, int *, int *, int *);
bool is_master(MPI_Comm);
}
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
static int little_endian() {
    const uint16_t value = 1;
    return *((const uint8_t *) &value) == 1;
}

static uint32_t swap32(uint32_t value) {
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8)  |
           ((value & 0x00ff0000u) >> 8)  |
           ((value & 0xff000000u) >> 24);
}

static void raw_path(const char *path, const char *name, char *out, size_t n) {
    size_t len = strlen(path);
    const char *suffix = ".raw";
    size_t suffix_len = strlen(suffix);
    int written;

    if (len >= suffix_len && strcmp(path + len - suffix_len, suffix) == 0)
        written = snprintf(out, n, "%.*s.%s.raw", (int) (len - suffix_len), path, name);
    else
        written = snprintf(out, n, "%s.%s.raw", path, name);
    if (written < 0 || (size_t) written >= n)
        ERR("raw grid path is too long");
}

static void raw_write_channel(const Coords *coords, MPI_Comm cart,
                              const char *path, const float *data) {
    int sizes[3] = { zdomain(coords), ydomain(coords), xdomain(coords) };
    int subsizes[3] = { zs(coords), ys(coords), xs(coords) };
    int starts[3] = { zlo(coords), ylo(coords), xlo(coords) };
    int local_count = subsizes[0] * subsizes[1] * subsizes[2];
    MPI_Datatype filetype;
    MPI_File file;
    MPI_Status status;
    float *swapped = NULL;
    const float *output = data;
    int i;
    int rc;
    char message[BUFSIZ];

    if (!little_endian()) {
        swapped = (float *) malloc((size_t) local_count * sizeof(*swapped));
        if (swapped == NULL) ERR("failed to allocate endian conversion buffer");
        for (i = 0; i < local_count; ++i) {
            uint32_t bits;
            memcpy(&bits, &data[i], sizeof(bits));
            bits = swap32(bits);
            memcpy(&swapped[i], &bits, sizeof(bits));
        }
        output = swapped;
    }

    rc = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                  MPI_FLOAT, &filetype);
    if (rc != MPI_SUCCESS) ERR("failed to create raw grid file view");
    rc = MPI_Type_commit(&filetype);
    if (rc != MPI_SUCCESS) ERR("failed to commit raw grid file view");

    rc = MPI_File_open(cart, path, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                       MPI_INFO_NULL, &file);
    if (rc != MPI_SUCCESS) ERR("failed to open raw grid file <%s>", path);
    rc = MPI_File_set_size(file, 0);
    if (rc != MPI_SUCCESS) ERR("failed to truncate raw grid file <%s>", path);
    rc = MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    if (rc != MPI_SUCCESS) ERR("failed to set raw grid file view <%s>", path);
    rc = MPI_File_write_all(file, output, local_count, MPI_FLOAT, &status);
    if (rc != MPI_SUCCESS) {
        MPI_Error_string(rc, message, &i);
        message[i] = '\0';
        ERR("failed to write raw grid file <%s>: %s", path, message);
    }
    rc = MPI_File_close(&file);
    if (rc != MPI_SUCCESS) ERR("failed to close raw grid file <%s>", path);
    rc = MPI_Type_free(&filetype);
    if (rc != MPI_SUCCESS) ERR("failed to free raw grid file view");
    free(swapped);
}

void raw_write(int3 N, MPI_Comm cart, const char *path, int ncomp,
               const float **data, const char **names) {
    char channel_path[FILENAME_MAX];
    Coords *coords;
    int i;

    UC(coords_ini(cart, N.x, N.y, N.z, &coords));
    for (i = 0; i < ncomp; ++i) {
        raw_path(path, names[i], channel_path, sizeof(channel_path));
        raw_write_channel(coords, cart, channel_path, data[i]);
    }
    UC(coords_fin(coords));
}




#include <string.h>
#include <stdio.h>
#include <vector_types.h>






void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names);

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
    float3 o, d;
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
static void xsuffix(const char *i,      char *o) {

    char s[] = ".raw";
    int n;
    n = strlen(i) - strlen(s);
    if (n >= 0 && eq(i + n, s)) {strncpy(o, i, n); o[n] = '\0';}
    else                        strcpy (o, i);
    strcat(o, ".xmf");
}
static void basename(const char *i,      char *o) {
    const char *p;
    p = i;
    while (*i != '\0') if (*i++ == '/') p = i;
    strcpy(o, p);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names) {
    char w[BUFSIZ];
    FILE *f;

    xsuffix(path,      w);
    UC(efopen(w, "w",      &f));
    header(f);

    basename(path,      w);
    grid(domainSize, gridSize, f, w, ncomp, names);
    epilogue(f);
    UC(efclose(f));
}
static int3 get_total_size(int3 L, MPI_Comm cart) {
    enum {X, Y, Z, D};
    int dims[3], periods[D], coords[D];
    int3 G;
    MC(m::Cart_get(cart, D, dims, periods, coords));
    G.x = L.x * dims[X];
    G.y = L.y * dims[Y];
    G.z = L.z * dims[Z];
    return G;
}

void grid_write(int3 N, int3 L, MPI_Comm cart, const char *path, int ncmp, const float **data, const char **names) {
    int3 domainSize, gridSize;

    domainSize = get_total_size(L, cart);
    gridSize   = get_total_size(N, cart);

    UC(raw_write(N, cart, path, ncmp, data, names));
    if (m::is_master(cart))
        UC(xmf_write(domainSize, gridSize, path, ncmp, names));
}
