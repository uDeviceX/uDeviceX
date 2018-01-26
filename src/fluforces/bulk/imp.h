struct BCloud {
    const float4 *pp;
    const int *cc;
};

void flocal(int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff);

