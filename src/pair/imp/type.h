struct PairParams {
    float a[MAX_PAR]; /* a    : conservative */
    float g[MAX_PAR]; /* gamma: dissipative  */
    float s[MAX_PAR]; /* sigma: random       */
    int ncolors;      /* number of colors    */

    float ljs; /* lennard jones sigma   */
    float lje; /* lennard jones epsilon */
};

