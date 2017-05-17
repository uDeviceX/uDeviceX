#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ply.h"

void read_infos(const char *fname, std::vector<float>& com, std::vector<float>& e1, std::vector<float>& e2, std::vector<float>& e3)
{
    FILE *f = fopen(fname, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Error: could not open <%s>\n", fname);
        exit(1);
    }

#define BSZ 2048 // max number of chars per line
#define xstr(s) str(s)
#define str(s) #s

    char line[BSZ+1] = {0};
    while(fscanf(f, " %[^\n]" xstr(BSZ) "c", line) == 1)
    {
        float r[3], r1[3], r2[3], r3[3];

#define DV "%*g %*g %*g " // [d]ummy [v]ector
#define V "%g %g %g "
#define dst(a) a, a + 1, a + 2

        const int nread = sscanf(line,
                                 /* t  com v fo to e0 e1 e2 fo to */
                                 "%*g " V DV DV DV V  V  V  DV DV,
                                 dst(r), dst(r1), dst(r2), dst(r3));
#undef DV
#undef V
#undef dst

        if (nread != 12)
        {
            fprintf(stderr, "%s : wrong format\n", fname);
            exit(0);
        }

#define pb(V, a) V.insert(V.end(), a, a + 3)
        pb(com, r);
        pb(e1, r1);
        pb(e2, r2);
        pb(e3, r3);
#undef pb
        
        memset(line, 0, sizeof(line));
    }
    fclose(f);
}

void concatenate(const std::vector<int>& ttn, const std::vector<float>& vvn,
                 std::vector<int>& tt, std::vector<float>& vv)
{
    const uint nv0 = vv.size() / 3;
    const uint nt0 = tt.size() / 3;

    vv.insert(vv.end(), vvn.begin(), vvn.end());
    tt.insert(tt.end(), ttn.begin(), ttn.end());

    for (uint i = 3*nt0; i < tt.size(); ++i)
    tt[i] += nv0;
}

void gen_vv(const int nv, const float *vv0, const float *com, const float *e1, const float *e2, const float *e3, float *vv)
{
    for (int i = 0; i < nv; ++i)
    {
        const float x = vv0[3*i + 0];
        const float y = vv0[3*i + 1];
        const float z = vv0[3*i + 2];
        float *r = vv + 3*i;

        r[0] = com[0] + x * e1[0] + y * e2[0] + z * e3[0];
        r[1] = com[1] + x * e1[1] + y * e2[1] + z * e3[1];
        r[2] = com[2] + x * e1[2] + y * e2[2] + z * e3[2];
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <mesh.ply> <solid_diag1.txt> <solid_diag2.txt> ...\n", argv[0]);
        exit(1);
    }

    const int ns = argc - 2;
    std::vector<std::vector<float>> coms(ns), e1s(ns), e2s(ns), e3s(ns);

    for (int i = 0; i < ns; ++i)
    read_infos(argv[2+i], coms[i], e1s[i], e2s[i], e3s[i]);
    
    std::vector<float> vv0, vv;
    std::vector<int> tt0, tt;

    read_ply(argv[1], tt0, vv0);

    
    
    return 0;
}
