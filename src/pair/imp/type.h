struct PairParams {
    float a[MAX_PAR]; /* a    : conservative */
    float g[MAX_PAR]; /* gamma: dissipative  */
    float s[MAX_PAR]; /* sigma: random       */
    int ncolors;      /* number of colors    */
    float spow;       /* s level             */

    float ljs; /* lennard jones sigma   */
    float lje; /* lennard jones epsilon */

    /* adhesion parameters */
    float k1; /* spring constant */ 
    float k2; /* viscous coeff   */
};

