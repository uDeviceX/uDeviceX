struct GridSampler;
struct GridSampleData;

struct Particle;

void grid_sampler_data_ini(GridSampleData **);
void grid_sampler_data_fin(GridSampleData  *);

void grid_sampler_data_reset(GridSampleData *);
void grid_sampler_data_push(long n, const Particle *pp, const float *ss, GridSampleData *);


void grid_sampler_ini(bool stress, int3 L, int3 N, GridSampler**);
void grid_sampler_fin(GridSampler*);

void grid_sampler_reset(GridSampler*);
void grid_sampler_add(const GridSampleData*, GridSampler*);
void grid_sampler_dump(MPI_Comm, const char *dir, long id, GridSampler*);
