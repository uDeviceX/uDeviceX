/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};


struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

/* mirrored */
struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};


struct PairPa {
    float x, y, z;
    float vx, vy, vz;
    /* optional fields */
    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;
    /* stress */
    float sxx, syy, szz, sxy, sxz, syz;
};
