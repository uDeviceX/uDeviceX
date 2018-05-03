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

// tag::mem[]
void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);
// end::mem[]

// tag::upd[]
void objects_clear_vel(Objects*);        // <1>
void objects_update(float dt, Objects*); // <2>
void objects_distribute(Objects*);       // <3>
// end::upd[]

// tag::force[]
void objects_clear_forces(Objects*);                 // <1>
void objects_internal_forces(float dt, Objects *o);  // <2>
void objects_body_forces(const BForce*, Objects *o); // <3>
// end::force[]

// tag::dump[]
void objects_mesh_dump(Objects*);                    // <1>
void objects_diag_dump(float t, Objects*);           // <2>
void objects_part_dump(long id, Objects*, IoBop*);

void objects_strt_templ(const char *base, Objects*);         // <3>
void objects_strt_dump(const char *base, long id, Objects*); // <4>
// end::dump[]

// tag::get[]
void objects_get_particles_all(Objects*, PFarrays*);    // <1>
void objects_get_particles_mbr(Objects*, PFarrays*);    // <2>
void objects_get_accel(const Objects*, TimeStepAccel*); // <3>
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
void objects_bounce(float dt, float flu_mass, const Clist flu_cells, PFarray *flu, Objects*); // <1>
void objects_recolor_flu(Objects*, PFarray *flu); // <2>
double objects_mbr_tot_volume(const Objects*);    // <3>
// end::tools[]

// tag::check[]
void objects_check_size(const Objects*);
void objects_check_vel(const Objects*, const Dbg*, float dt);
void objects_check_forces(const Objects*, const Dbg*, float dt);
// end::check[]

