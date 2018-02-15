enum {
    BLUE,
    RED,
    MAX_COL
}; 

/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (MAX_COL * (MAX_COL + 1)) / 2
};


struct PairDPD {
    float a, g, s;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR];
};

/* mirrored */
struct PairDPDCM {
    int ncolors;
    float a[MAX_COL], g[MAX_COL], s[MAX_COL];
};

struct PairDPDLJ {
    float a, g, s;
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
