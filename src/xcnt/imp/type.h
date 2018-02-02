struct Contact {
    Clist cells;    /* cell lists              */
    ClistMap *cmap; /* helper to build clists  */
    RNDunif *rgen;  /* random number generator */
    int3 L;         /* dimensions of subdomain */
};
