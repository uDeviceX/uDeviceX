#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
namespace mpicheck { void check(int, const char *, int); }
#include <stdlib.h>
#include <mpi.h>




namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);


bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct WriteFile;

int write_file_open(MPI_Comm, const char *path,      WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*);
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*);

int write_shift_indices(MPI_Comm, int,      int*);
int write_reduce(MPI_Comm, int,      int*);
struct WriteFile {
    MPI_File f;
};
int write_file_open(MPI_Comm comm, const char *fn,      WriteFile **pq) {
    WriteFile *q;
    EMALLOC(1, &q);
    MC(m::File_open(comm, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &q->f));
    MC(m::File_set_size(q->f, 0));
    *pq = q;
    return 0;
}

int write_file_close(WriteFile *fp) {
    MPI_File f;
    f = fp->f;
    MC(m::File_close(&f));
    EFREE(fp);
    return 0;
}

void write_all(MPI_Comm comm, const void * const ptr, int nbytes32, WriteFile *fp) {
    MPI_File f = fp->f;
    MPI_Offset base;
    MPI_Offset offset = 0, nbytes = nbytes32;
    MPI_Status status;
    MPI_Offset ntotal = 0;
    MC(m::File_get_position(f, &base));
    MC(m::Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));
    MC(m::File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MC(m::Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );
    MC(m::File_seek(f, ntotal, MPI_SEEK_CUR));
}

int write_master(MPI_Comm comm, const void * const ptr, int sz0, WriteFile *fp) {
    int sz;
    sz = m::is_master(comm) ? sz0 : 0;
    write_all(comm, ptr, sz, fp);
    return 0;
}

int write_shift_indices(MPI_Comm comm, int n, int *shift0) {
    *shift0 = 0;
    MC(m::Exscan(&n, shift0, 1, MPI_INTEGER, MPI_SUM, comm));
    return 0;
}

int write_reduce(MPI_Comm comm, int n0,      int *n) {
    *n = 0;
    MC(m::Reduce(&n0, n, 1, MPI_INT, MPI_SUM, 0, comm));
    return 0;
}
