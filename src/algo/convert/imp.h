struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp, /**/ float4 *zpp);
