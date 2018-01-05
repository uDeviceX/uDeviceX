struct Particle;
// tag::interface[]
void minmax(const Particle * const particles, int nparticles_per_body, int nbodies,
            float3 * minextents, float3 * maxextents);
// end::interface[]
