struct Wvel;
struct Wvel_v;

struct float3;
struct Config;

// tag::mem[]
void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);
// end::ini[]

// tag::ini[]
void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, int log_freq, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
// end::ini[]

// tag::cnf[]
void wvel_set_conf(const Config *cfg, Wvel *vw);
// end::cnf[]

// tag::int[]
void wvel_get_view(long it, const Wvel *wv, /**/ Wvel_v *view);
// end::int[]
