struct Objects;

struct Sdf;
struct Config;
struct Coords;
struct Opt;
struct PFarray;
struct PFarrays;
struct BForce;
struct TimeStepAccel;
struct Dbg;
struct IoBop;
struct PairParams;
struct WallRepulsePrm;

// tag::mem[]
void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);
// end::mem[]

void objects_save_mesh(Objects*);
// tag::upd[]
void objects_clear_vel(Objects*);         // <1>
void objects_advance(float dt, Objects*); // <2>
void objects_distribute(Objects*);        // <3>
void objects_update_dpd_prms(float dt, float kBT, Objects*); // <4>
// end::upd[]

// tag::force[]
void objects_clear_forces(Objects*);                 // <1>
void objects_body_forces(const BForce*, Objects *o); // <2>
// end::force[]

// tag::dump[]
void objects_mesh_dump(Objects*);                    // <1>
void objects_diag_dump(float t, Objects*);           // <2>
void objects_part_dump(long id, Objects*, IoBop*);   // <3>

void objects_strt_templ(const char *base, Objects*);         // <4>
void objects_strt_dump(const char *base, long id, Objects*); // <5>
// end::dump[]

// tag::get[]
bool objects_have_bounce(const Objects*);

void objects_get_particles_all(Objects*, PFarrays*);    // <1>
void objects_get_particles_mbr(Objects*, PFarrays*);    // <2>
void objects_get_accel(const Objects*, TimeStepAccel*); // <3>

void objects_get_params_fsi(const Objects*, const PairParams*[]);           // <4>
void objects_get_params_adhesion(const Objects*, const PairParams*[]);      // <5>
void objects_get_params_repulsion(const Objects*, const WallRepulsePrm*[]); // <6>
// end::get[]

// tag::gen[]
void objects_gen_mesh(Objects*);                           // <1>
void objects_remove_from_wall(const Sdf *sdf, Objects *o); // <2>
void objects_gen_freeze(PFarray*, Objects*);               // <3>
// end::gen[]

// tag::strt[]
void objects_restart(Objects*);
// end::strt[]

// tag::tools[]
void objects_bounce(float dt, float flu_mass, const Clist flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj);
void objects_recolor_flu(Objects*, PFarray *flu); // <2>
double objects_mbr_tot_volume(const Objects*);    // <3>
// end::tools[]

// tag::check[]
void objects_check_size(const Objects*);
void objects_check_vel(const Objects*, const Dbg*, float dt);
void objects_check_forces(const Objects*, const Dbg*, float dt);
// end::check[]

