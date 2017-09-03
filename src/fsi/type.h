namespace fsi {
struct SolventWrap {
    Particle *pp;
    hforces::Cloud c;
    Force *ff;
    int n; 
    int *starts;
};
}
