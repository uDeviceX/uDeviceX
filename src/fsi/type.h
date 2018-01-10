struct RNDunif;
namespace fsi {
struct SolventWrap {
    Cloud c;
    Force *ff;
    int n; 
    int *starts;
};

struct Fsi {
    SolventWrap *wo;
    RNDunif     *rgen;
};
}
