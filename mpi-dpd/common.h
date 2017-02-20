#ifdef NDEBUG
#define cuda_printf(...)
#else
#define cuda_printf(...) do { printf(__VA_ARGS__); } while(0)
#endif

const float dt            = _dt;
const float rbc_mass      = _rbc_mass;
const float gamma_dot     = _gamma_dot;
const float hydrostatic_a = _hydrostatic_a / rc;
const float kBT           = _kBT           / (rc*rc);
const int numberdensity   = _numberdensity * (rc*rc*rc);

extern float tend;
extern bool walls, pushtheflow, doublepoiseuille, rbcs, hdf5field_dumps, hdf5part_dumps, is_mps_enabled, contactforces;
extern int steps_per_dump, steps_per_hdf5dump, wall_creation_stepid;

/* [c]cuda [c]heck */
#define CC(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    abort();
  }
}

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)
inline void mpiAssert(int code, const char *file, int line)
{
  if (code != MPI_SUCCESS)
  {
    char error_string[2048];
    int length_of_error_string = sizeof(error_string);
    MPI_Error_string(code, error_string, &length_of_error_string);

    printf("mpiAssert: %s %d %s\n", file, line, error_string);

    MPI_Abort(MPI_COMM_WORLD, code);
  }
}

//AoS is the currency for dpd simulations (because of the spatial locality).
//AoS - SoA conversion might be performed within the hpc kernels.
struct Particle
{
  float x[3], u[3];

  static bool initialized;
  static MPI_Datatype mytype;

  static MPI_Datatype datatype()
  {
    if (!initialized)
    {
      MPI_CHECK( MPI_Type_contiguous(6, MPI_FLOAT, &mytype));

      MPI_CHECK(MPI_Type_commit(&mytype));

      initialized = true;
    }

    return mytype;
  }
};

struct Acceleration
{
  float a[3];

  static bool initialized;
  static MPI_Datatype mytype;

  static MPI_Datatype datatype()
  {
    if (!initialized)
    {
      MPI_CHECK( MPI_Type_contiguous(3, MPI_FLOAT, &mytype));

      MPI_CHECK(MPI_Type_commit(&mytype));

      initialized = true;
    }

    return mytype;
  }
};

struct ParticlesWrap
{
  const Particle * p;
  Acceleration * a;
  int n;

  ParticlesWrap() : p(NULL), a(NULL), n(0){}

  ParticlesWrap(const Particle * const p, const int n, Acceleration * a):
      p(p), n(n), a(a) {}
};

struct SolventWrap : ParticlesWrap
{
  const int * cellsstart, * cellscount;

  SolventWrap(): cellsstart(NULL), cellscount(NULL), ParticlesWrap() {}

  SolventWrap(const Particle * const p, const int n, Acceleration * a, const int * const cellsstart, const int * const cellscount):
      ParticlesWrap(p, n, a), cellsstart(cellsstart), cellscount(cellscount) {}
};

/* container for the gpu particles during the simulation */
template<typename T>
struct SimpleDeviceBuffer {
  int capacity, S;    /* `S' is for size */
  T* D;               /* `D' is for data */
  explicit SimpleDeviceBuffer(int n = 0): capacity(0), S(0), D(NULL) { resize(n);}
  ~SimpleDeviceBuffer() {
    if (D != NULL) CC(cudaFree(D));
    D = NULL;
  }

  void dispose() {
    if (D != NULL) CC(cudaFree(D));
    D = NULL;
  }

  void resize(int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL)  CC(cudaFree(D));
    int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * capacity));
  }

  void preserve_resize(int n) {
    T * old = D;
    int oldsize = S;

    S = n;
    if (capacity >= n) return;
    int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * capacity));
    if (old != NULL) {
      CC(cudaMemcpy(D, old, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice));
      CC(cudaFree(old));
    }
  }
};

template<typename T>
struct SimpleHostBuffer
{
  int capacity, size;

  T * data;

  explicit SimpleHostBuffer(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}

  ~SimpleHostBuffer()
  {
    if (data != NULL)
      CC(cudaFreeHost(data));

    data = NULL;
  }

  void resize(const int n)
  {
    size = n;

    if (capacity >= n)
      return;

    if (data != NULL)
      CC(cudaFreeHost(data));

    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocDefault));
  }

  void preserve_resize(const int n)
  {
    T * old = data;

    const int oldsize = size;

    size = n;

    if (capacity >= n)
      return;

    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    data = NULL;
    CC(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocDefault));

    if (old != NULL)
    {
      memcpy(data, old, sizeof(T) * oldsize);
      CC(cudaFreeHost(old));
    }
  }
};

template<typename T>
struct PinnedHostBuffer
{
  int capacity, size;

  T * data, * devptr;

  PinnedHostBuffer(int n = 0): capacity(0), size(0), data(NULL), devptr(NULL) { resize(n);}

  ~PinnedHostBuffer()
  {
    if (data != NULL)
      CC(cudaFreeHost(data));

    data = NULL;
  }

  void resize(const int n)
  {
    size = n;

    if (capacity >= n)
      return;

    if (data != NULL)
      CC(cudaFreeHost(data));

    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&devptr, data, 0));
  }

  void preserve_resize(const int n)
  {
    T * old = data;

    const int oldsize = size;

    size = n;

    if (capacity >= n)
      return;

    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    data = NULL;
    CC(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

    if (old != NULL)
    {
      CC(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyHostToHost));
      CC(cudaFreeHost(old));
    }

    CC(cudaHostGetDevicePointer(&devptr, data, 0));
  }
};

/* container for the cell lists, which contains only two integer
vectors of size ncells.  the start[cell-id] array gives the entry in
the particle array associated to first particle belonging to cell-id
count[cell-id] tells how many particles are inside cell-id.  building
the cell lists involve a reordering of the particle array (!) */
struct CellLists
{
  const int ncells, LX, LY, LZ;

  int * start, * count;

  CellLists(const int LX, const int LY, const int LZ):
      ncells(LX * LY * LZ + 1), LX(LX), LY(LY), LZ(LZ)
  {
    CC(cudaMalloc(&start, sizeof(int) * ncells));
    CC(cudaMalloc(&count, sizeof(int) * ncells));
  }

  void build(Particle * const p, const int n, cudaStream_t stream, int * const order = NULL, const Particle * const src = NULL);

  ~CellLists()
  {
    CC(cudaFree(start));
    CC(cudaFree(count));
  }
};

struct ExpectedMessageSizes
{
  int msgsizes[27];
};

void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle * _particles, int n, float dt, int idstep, Acceleration * _acc);

class LocalComm
{
  MPI_Comm local_comm, active_comm;
  int local_rank, local_nranks;
  int rank, nranks;

  char name[MPI_MAX_PROCESSOR_NAME];
  int len;

 public:
  LocalComm();

  void initialize(MPI_Comm active_comm);

  void barrier();

  void print_particles(int np);

  int get_size() { return local_nranks; }

  int get_rank() { return local_rank; }

  MPI_Comm get_comm() { return local_comm;  }
};
