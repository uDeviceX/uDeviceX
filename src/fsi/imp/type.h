struct RNDunif;

struct SolventWrap {
    PaArray pa;
    Force *ff;
    int n; 
    int *starts;
};

struct Fsi {
    SolventWrap *wo;
    RNDunif     *rgen;
    int3 L; /* subdomain size */
};
