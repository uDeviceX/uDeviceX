enum {
    BLUE,
    RED,
    MAX_COL
}; 

/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (MAX_COL * (MAX_COL + 1)) / 2
};

struct PairParams {
    float a[MAX_PAR]; /* a    : conservative */
    float g[MAX_PAR]; /* gamma: dissipative  */
    float s[MAX_PAR]; /* sigma: random       */
    int ncol;         /* number of colors    */

    float lj_s; /* lennard jones sigma   */
    float lj_e; /* lennard jones epsilon */
};

