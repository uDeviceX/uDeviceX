struct BPaArray_v {
    const float4 *pp;
};

struct BPaCArray_v {
    const float4 *pp;
    const int *cc;
};

struct TBPaArray_v {
    Texo<float4> pp;
};

struct TBPaCArray_v {
    Texo<float4> pp;
    Texo<int> cc;
};
