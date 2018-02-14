struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;

struct BPaArray {
    const float4 *pp;
    const int *cc;
};

void flocal      (const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff);
void flocal_color(const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff);

