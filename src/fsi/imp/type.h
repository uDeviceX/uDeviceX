#pragma once

struct RNDunif;

struct FsiSolventWrap {
    PaArray pa;
    Force *ff;
    int n; 
    const int *starts;
};

struct Fsi {
    FsiSolventWrap *wo;
    RNDunif     *rgen;
    int3 L; /* subdomain size */
};
