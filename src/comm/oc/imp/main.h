#define MAX_CONTEXT 10
static char oc_file[BUFSIZ];
static int  oc_line;

static char mpi_file[BUFSIZ];
static char mpi_err[BUFSIZ];
static int  mpi_line;
static int  mpi_status;

static char context[MAX_CONTEXT][BUFSIZ];
static int  n_context;

void before(const char *file, int line) {
    n_context = mpi_status = 0;
    strcpy(oc_file, file);
    oc_line = line;
}
int  error(int rc) { return rc || mpi_status; }
void report() {
    int i;
    MSG("<< comm error");
    MSG("%s:%d:", oc_file, oc_line);
    if (mpi_status != 0) MSG("%s:%d: %s", mpi_file, mpi_line, mpi_err);
    if (n_context > 0)   MSG("context:");
    for (i = 0; i < n_context; i++) MSG("%s", context[n_context]);
    ERR(">> comm error");
}

int status() { return mpi_status; }
void mpi_check(int code, const char *file, int line) {
    int n;
    if (code == MPI_SUCCESS) return;
    MPI_Error_string(code, /**/ mpi_err, &n); mpi_err[n + 1] = '\n';
    mpi_status = 1;
    strcpy(mpi_file, file);
    mpi_line = line;
}
