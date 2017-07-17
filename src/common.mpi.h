/* [m]pi [c]heck */
#define MC(ans)                                             \
    do { mpiAssert((ans), __FILE__, __LINE__); } while (0)
inline void mpiAssert(int code, const char *file, int line) {
    if (code != MPI_SUCCESS) {
        char error_string[2048];
        int length_of_error_string = sizeof(error_string);
        MPI_Error_string(code, error_string, &length_of_error_string);
        printf("mpiAssert: %s %d %s\n", file, line, error_string);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}

namespace datatype {
MPI_Datatype particle, solid;

void ini() {
    MC(MPI_Type_contiguous(6, MPI_FLOAT, &particle));
    MC(MPI_Type_contiguous(32, MPI_FLOAT, &solid));

    MC(MPI_Type_commit(&particle));
    MC(MPI_Type_commit(&solid));
}

void fin() {
    MC(MPI_Type_free(&particle));
    MC(MPI_Type_free(&solid));
}
}
