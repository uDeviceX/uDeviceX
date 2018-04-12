struct Sampler;
struct Particle;

struct SampleData {
    long n;
    const Particle *pp;
    const float *ss;
}

void sampler_ini(Sampler**);
void sampler_fin(Sampler*);

void sampler_reset(Sampler*);
void sampler_add(const SampleData, Sampler*);
void sampler_dump(Sampler*);
