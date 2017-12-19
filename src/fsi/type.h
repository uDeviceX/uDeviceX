namespace fsi {
struct SolventWrap {
    Particle *pp;
    Cloud c;
    Force *ff;
    int n; 
    int *starts;
};

struct Fsi {
    SolventWrap* wo;
    RNDunif* rgen;
};
}
