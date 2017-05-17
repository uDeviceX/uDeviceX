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

        /*              t  com v fo to e0 e1 e2 fo to */
        sscanf(line, "%*g " V DV DV DV V  V  V  DV DV,
               dst(r), dst(r1), dst(r2), dst(r3));
#undef DV
#undef V
#undef dst

#define pb(V, a) do {                           \
            V.push_back(a[0]);                  \
            V.push_back(a[1]);                  \
            V.push_back(a[2]);                  \
        } while(0)

        pb(com, r);
        pb(e1, r1);
        pb(e2, r2);
        pb(e3, r3);
#undef pb
        
        memset(line, 0, sizeof(line));
    }
    fclose(f);
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
