void flocal(const float4 *zip0, const ushort4 *zip1, int n,
            const int *start, const int *count, rnd::KISS *rnd, /**/ Force *ff);
void flocal_color(const float4 *zip0, const ushort4 *zip1, const int *colors,
                  int n, const int *start, const int *count, rnd::KISS* rnd, /**/ Force *ff);
