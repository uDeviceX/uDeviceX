struct float4;
struct int3;
struct BCloud {
    const float4 *pp;
    const int *cc;
};

void flocal(int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff);

