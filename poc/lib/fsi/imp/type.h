struct RNDunif;

struct SolventWrap {
    PaArray pa;
    Force *ff;
    int n; 
    const int *starts;
};

struct Fsi {
    SolventWrap *wo;
    RNDunif     *rgen;
    int3 L; /* subdomain size */
};
