struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;

struct BCloud {
    const float4 *pp;
    const int *cc;
};

void flocal       (const PairParams*, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff);
void flocal_colors(const PairParams*, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff);

