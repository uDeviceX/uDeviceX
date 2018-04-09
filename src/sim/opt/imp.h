struct Opt {
    bool fsi, cnt;
    bool flucolors, fluids, fluss;
    bool rbc, rbcids, rbcstretch;
    int rbcshifttype;
    bool rig, rig_bounce, rig_empty_pp;
    int rigshifttype;
    bool wall;
    bool inflow, outflow, denoutflow, vcon;
    bool dump_field, dump_parts, dump_strt, dump_rbc_com, dump_forces;
    float freq_field, freq_parts, freq_strt, freq_rbc_com;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
    int recolor_freq;
    bool push_flu, push_rbc, push_rig;
};

struct Config;

void opt_read_gen(const Config*, Opt*);
void opt_read_full(const Config*, Opt*);
void opt_check(const Opt*);
