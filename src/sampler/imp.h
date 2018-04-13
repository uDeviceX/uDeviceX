struct GridSampler;
struct GridSampleData;

struct Particle;

void sampler_data_ini(GridSampleData **);
void sampler_data_fin(GridSampleData  *);

void sampler_data_reset(GridSampleData *);
void sampler_data_push(long n, const Particle *pp, const float *ss, GridSampleData *);


void sampler_ini(bool stress, int3 L, int3 N, GridSampler**);
void sampler_fin(GridSampler*);

void sampler_reset(GridSampler*);
void sampler_add(const GridSampleData*, GridSampler*);
void sampler_dump(MPI_Comm, const char *dir, long id, GridSampler*);
