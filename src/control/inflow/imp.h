struct Inflow;

void ini(int2 nc, Inflow **i);
void fin(Inflow *i);

void ini_params_plate(float3 o, float3 a, float3 b,
                      float3 u, bool upoiseuille, bool vpoiseuille,
                      /**/ Inflow *i);


void create_pp(Inflow *i, int *n, Particle *pp);


