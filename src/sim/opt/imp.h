/* membrane options */
struct ObpMbr {
    bool active, ids, stretch, push;
    int shifttype;
};

struct OptRig {
    bool active, bounce, empty_pp, push;
    int shifttype;
};

struct Opt {
    ObpMbr rbc;
    OptRig rig;
    bool fsi, cnt;
    bool flucolors, fluids, fluss;
    bool wall;
    bool inflow, outflow, denoutflow, vcon;
    bool dump_field, dump_parts, dump_strt, dump_rbc_com, dump_forces;
    float freq_field, freq_parts, freq_strt, freq_rbc_com;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
    int recolor_freq;
    bool push_flu;
    int sampler_npdump;
    int3 sampler_grid_ref;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
