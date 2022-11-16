struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;
struct FoArray;

struct BPaArray {
    bool colors;
    const float4 *pp;
    const int *cc;
};

void flocal_push_pp(const float4 *pp, BPaArray *a);
void flocal_push_cc(const int *cc, BPaArray *a);

void flocal_apply(const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,
                  /**/ const FoArray *ff);

