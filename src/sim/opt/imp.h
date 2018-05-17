/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* membrane options */
struct OptMbr {
    bool active, ids, stretch, push, dump_com;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig {
    bool active, bounce, empty_pp, push;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
};

struct OptWall {
    bool active;
    bool repulse;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr rbc;
    OptRig rig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool fsi, cnt;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
