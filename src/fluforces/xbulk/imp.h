struct BCloud {
    const float4 *pp;
    const int *cc;
};

void flocal(int n, BCloud cloud, const int *start, const int *count, RNDunif *rnd, /**/ Force *ff);

