const float dt            = _dt;
const float rbc_mass      = _rbc_mass;
const float gamma_dot     = _gamma_dot;
const float hydrostatic_a = _hydrostatic_a / rc;
const float kBT           = _kBT / (rc * rc);
const int   numberdensity = _numberdensity * (rc * rc * rc);

extern float tend;
extern bool walls, pushtheflow, doublepoiseuille, rbcs, hdf5field_dumps,
  hdf5part_dumps, contactforces;
extern int steps_per_dump, steps_per_hdf5dump, wall_creation_stepid;

/* [c]cuda [c]heck */
#define CC(ans)							\
  do { cudaAssert((ans), __FILE__, __LINE__); } while (0)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
	    line);
    abort();
  }
}

/* [m]pi [c]heck */
#define MC(ans)							\
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

// AoS is the currency for dpd simulations (because of the spatial locality).
// AoS - SoA conversion might be performed within the hpc kernels.
struct Particle {
  float r[3], v[3];
  static bool initialized;
  static MPI_Datatype mytype;
  static MPI_Datatype datatype() {
    if (!initialized) {
      MC(MPI_Type_contiguous(6, MPI_FLOAT, &mytype));
      MC(MPI_Type_commit(&mytype));
      initialized = true;
    }
    return mytype;
  }
};

struct Force {
  float a[3];
};

struct ParticlesWrap {
  const Particle *p;
  Force *a;
  int n;
  ParticlesWrap() : p(NULL), a(NULL), n(0) {}
  ParticlesWrap(const Particle *const p, const int n, Force *a)
    : p(p), n(n), a(a) {}
};

struct SolventWrap : ParticlesWrap {
  const int *cellsstart, *cellscount;
  explicit SolventWrap() : cellsstart(NULL), cellscount(NULL), ParticlesWrap() {}
  SolventWrap(const Particle *const p, const int n, Force *a,
	      const int *const cellsstart, const int *const cellscount)
    : ParticlesWrap(p, n, a),
      cellsstart(cellsstart),
      cellscount(cellscount) {}
};

/* container for the gpu particles during the simulation */
template <typename T> struct DeviceBuffer {
  typedef T value_type;
  /* `C': capacity; `S': size; `D' : data*/
  int C, S; T *D;

  explicit DeviceBuffer(int n = 0) : C(0), S(0), D(NULL) { resize(n); }
  ~DeviceBuffer() {
    if (D != NULL) CC(cudaFree(D));
    D = NULL;
  }

  void resize(int n) {
    S = n;
    if (C >= n) return;
    if (D != NULL) CC(cudaFree(D));
    int conservative_estimate = (int)ceil(1.1 * n);
    C = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * C));
  }

  void preserve_resize(int n) {
    T *old = D;
    int oldS = S;

    S = n;
    if (C >= n) return;
    int conservative_estimate = (int)ceil(1.1 * n);
    C = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * C));
    if (old != NULL) {
      CC(cudaMemcpy(D, old, sizeof(T) * oldS, cudaMemcpyDeviceToDevice));
      CC(cudaFree(old));
    }
  }
};

template <typename T> struct PinnedHostBuffer {
private:
  int capacity;
public:
  typedef T value_type;
  /* `S': size; `D' is for data; `DP' device pointer */
  int S;  T *D, *DP;

  explicit PinnedHostBuffer(int n = 0)
    : capacity(0), S(0), D(NULL), DP(NULL) {
    resize(n);
  }

  ~PinnedHostBuffer() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }

  void resize(const int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL) CC(cudaFreeHost(D));
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }

  void preserve_resize(const int n) {
    T *old = D;
    const int oldS = S;
    S = n;
    if (capacity >= n) return;
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);
    D = NULL;
    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));
    if (old != NULL) {
      CC(cudaMemcpy(D, old, sizeof(T) * oldS, cudaMemcpyHostToHost));
      CC(cudaFreeHost(old));
    }
    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }
};

/* container for the cell lists, which contains only two integer
   vectors of size ncells.  the start[cell-id] array gives the entry in
   the particle array associated to first particle belonging to cell-id
   count[cell-id] tells how many particles are inside cell-id.  building
   the cell lists involve a reordering of the particle array (!) */
struct CellLists {
  const int ncells, LX, LY, LZ;
  int *start, *count;
  CellLists(const int LX, const int LY, const int LZ)
    : ncells(LX * LY * LZ + 1), LX(LX), LY(LY), LZ(LZ) {
    CC(cudaMalloc(&start, sizeof(int) * ncells));
    CC(cudaMalloc(&count, sizeof(int) * ncells));
  }

  void build(Particle *const p, const int n,
	     int *const order = NULL, const Particle *const src = NULL);

  ~CellLists() {
    CC(cudaFree(start));
    CC(cudaFree(count));
  }
};

struct ExpectedMessageSizes {
  int msgsizes[27];
};

void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle *_particles, int n,
		 float dt, int idstep);
