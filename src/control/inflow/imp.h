struct Inflow;

void ini(int2 nc, Inflow **i);
void ini_velocity(Inflow *i);

void ini_params_plate(Coords c, float3 o, int dir, float L1, float L2,
                      float3 u, bool upoiseuille, bool vpoiseuille,
                      /**/ Inflow *i);
void ini_params_circle(float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i);

void fin(Inflow *i);

void create_pp(Inflow *i, int *n, Particle *pp);


