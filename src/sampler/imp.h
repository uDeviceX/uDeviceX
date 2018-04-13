struct Sampler;
struct SampleData;

struct Particle;

void sampler_data_ini(SampleData **);
void sampler_data_fin(SampleData  *);

void sampler_data_reset(SampleData *);
void sampler_data_push(long n, const Particle *pp, const float *ss, SampleData *);


void sampler_ini(bool stress, int3 L, int3 N, Sampler**);
void sampler_fin(Sampler*);

void sampler_reset(Sampler*);
void sampler_add(const SampleData*, Sampler*);
void sampler_dump(Sampler*);
