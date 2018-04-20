/* membrane options */
struct OptMbr {
    bool active, ids, stretch, push, dump_com;
    int shifttype;
    float freq_com;
};

/* rigid options */
struct OptRig {
    bool active, bounce, empty_pp, push;
    int shifttype;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct Opt {
    OptMbr rbc;
    OptRig rig;
    OptParams params;
    bool fsi, cnt;
    bool flucolors, fluids, fluss;
    bool wall;
    bool inflow, outflow, denoutflow, vcon;
    bool dump_field, dump_parts, dump_strt, dump_forces;
    float freq_field, freq_parts, freq_strt;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
    int recolor_freq;
    bool push_flu;
    int sampler_npdump;
    int3 sampler_grid_ref;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
