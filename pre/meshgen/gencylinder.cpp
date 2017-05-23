#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "ply.h"
#include "mesh.h"

using std::vector;

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage : %s <out.ply> <R> <nsubR> <H> <nsubH>\n", argv[0]);
        exit(1);
    }
    
    vector<int> tt;
    vector<float> vv;
    
    const float R = atof(argv[2]);
    const int  nR = atoi(argv[3]);
    const float H = atof(argv[4]);
    const int  nH = atoi(argv[5]);

    const float dt = (2 * M_PI) / nR;
    const float dz = H / nH;

    
    auto add_element = [&] (const float t, const float z)
    {
        const double a[3] = {R * cos(t)     , R * sin(t)     , z     };
        const double b[3] = {R * cos(t + dt), R * sin(t + dt), z     };
        const double c[3] = {R * cos(t)     , R * sin(t)     , z + dz};
        const double d[3] = {R * cos(t + dt), R * sin(t + dt), z + dz};

        const int offset = vv.size() / 3;

#define pb(v) do{vv.push_back(v[0]); vv.push_back(v[1]); vv.push_back(v[2]);} while(0)

        pb(a); pb(b); pb(c); pb(d);

#undef  pb
#define pb(t1, t2, t3) do{tt.push_back(t1); tt.push_back(t2); tt.push_back(t3);} while(0)

        pb(offset + 0, offset + 1, offset + 2); // abc
        pb(offset + 1, offset + 3, offset + 2); // bdc

#undef  pb
    };

    auto add_column = [&] (const float t)
    {
        for (int iz = 0; iz < nH; ++iz)
        {
            const float z = -H/2 + iz * dz;
            add_element(t, z);
        }
    };

    for (int it = 0; it < nR; ++it)
    add_column(it * dt);

    // reorder, remove duplicates
    {
        vector<float3> aos;
        soa2aos(tt, vv, aos);
        aos2soa(aos, tt, vv);
    }
    
    write_ply(argv[1], tt, vv);
    
    return 0;
}
