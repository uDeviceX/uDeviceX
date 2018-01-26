enum {
    BLUE,
    RED,
    MAX_COL
}; 

/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (MAX_COL * (MAX_COL + 1)) / 2
};

struct PairLJ {
    float s, e;
};

struct PairDPD {
    float a, g, s;
};

struct PairDPDC {
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR];
};

