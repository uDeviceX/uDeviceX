// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/grid/raw/imp.cpp */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>

struct Coords;
void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c);
void coords_fin(Coords *c);
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                           \
    } while (0)

#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();
void error_signal(const char *file, int line, const char *fmt, ...);
bool error_get();
void error_report();
void error_abort();

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
/* end src/io/grid/raw/imp.cpp */
