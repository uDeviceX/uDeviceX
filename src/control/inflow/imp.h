struct Inflow;

// tag::mem[]
void inflow_ini(int2 nc, Inflow **i);
void inflow_fin(Inflow *i);
// end::mem[]

// tag::ini[]
void inflow_ini_velocity(Inflow *i);

void inflow_ini_params_plate(Coords c, float3 o, int dir, float L1, float L2,
                      float3 u, bool upoiseuille, bool vpoiseuille,
                      /**/ Inflow *i);
void inflow_ini_params_circle(Coords c, float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i);
// end::ini[]

// tag::int[]
void inflow_create_pp(Inflow *i, int *n, Particle *pp);
// end::int[]
