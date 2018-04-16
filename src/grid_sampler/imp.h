struct GridSampler;
struct GridSampleData;

struct Particle;

// tag::data_mem[]
void grid_sampler_data_ini(GridSampleData **);
void grid_sampler_data_fin(GridSampleData  *);
// end::data_mem[]

// tag::data_int[]
void grid_sampler_data_reset(GridSampleData *); // <1>
void grid_sampler_data_push(long n, const Particle *pp, const float *ss, GridSampleData *); // <2>
// end::data_int[]


// tag::mem[]
void grid_sampler_ini(bool stress, int3 L, int3 N, GridSampler**); // <1>
void grid_sampler_fin(GridSampler*); // <2>
// end::mem[]

// tag::int[]
void grid_sampler_reset(GridSampler*); // <1>
void grid_sampler_add(const GridSampleData*, GridSampler*); // <2>
void grid_sampler_dump(MPI_Comm, const char *dir, long id, GridSampler*); // <3>
// end::int[]
