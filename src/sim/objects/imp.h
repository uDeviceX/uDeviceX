struct Objects;

struct Sdf;
struct Config;
struct Coords;
struct Opt;
struct PFarray;
struct PFarrays;
struct BForce;

void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);
    
void objects_clear_vel(Objects*);
void objects_update(float dt, Objects*);
void objects_distribute(Objects*);

void objects_clear_forces(Objects*);
void objects_internal_forces(float dt, Objects *o);
void objects_body_forces(const BForce*, Objects *o);

void objects_mesh_dump(Objects*);

void objects_strt_templ(const char *base, Objects*);
void objects_strt_dump(const char *base, long id, Objects*);

void objects_get_particles_all(Objects*, PFarrays*);
void objects_get_particles_mbr(Objects*, PFarrays*);

void objects_gen_mesh(Objects*);
void objects_remove_from_wall(const Sdf *sdf, Objects *o);
void objects_gen_freeze(PFarray*, Objects*);

void objects_restart(Objects*);

void objects_bounce(float dt, float flu_mass, const Clist flu_cells, PFarray *flu, Objects*);

// TODO
void objects_recolor_flu(Objects*, PFarrays *flu);
