#include "mesh.h"
#include <cmath>

using std::vector;

void icosahedron(vector<int>& tt, vector<float>& vv)
{
    const float phi = 0.5 * (1. + sqrt(5.));
    
    tt.clear(); vv.clear();

#define pp push_back
#define pp3(VV, a, b, c) do{VV.pp(a); VV.pp(b); VV.pp(c);} while(0)
    
    pp3(vv, -1, phi ,0);
    pp3(vv, 1, phi, 0);
    pp3(vv, -1, -phi, 0);
    pp3(vv, 1, -phi, 0);

    pp3(vv, 0, -1, phi);
    pp3(vv, 0, 1, phi);
    pp3(vv, 0, -1, -phi);
    pp3(vv, 0, 1, -phi);

    pp3(vv, phi, 0, -1);
    pp3(vv, phi, 0, 1);
    pp3(vv, -phi, 0, -1);
    pp3(vv, -phi, 0, 1);

    pp3(tt, 0, 11, 5);
    pp3(tt, 0, 5, 1);
    pp3(tt, 0, 1, 7);
    pp3(tt, 0, 7, 10);
    pp3(tt, 0, 10, 11);
    
    pp3(tt, 1, 5, 9);
    pp3(tt, 5, 11, 4);
    pp3(tt, 11, 10, 2);
    pp3(tt, 10, 7, 6);
    pp3(tt, 7, 1, 8);
    
    pp3(tt, 3, 9, 4);
    pp3(tt, 3, 4, 2);
    pp3(tt, 3, 2, 6);
    pp3(tt, 3, 6, 8);
    pp3(tt, 3, 8, 9);
    
    pp3(tt, 4, 9, 5);
    pp3(tt, 2, 4, 11);
    pp3(tt, 6, 2, 10);
    pp3(tt, 8, 6, 7);
    pp3(tt, 9, 8, 1);

#undef pp3
#undef pp
}

void soa2aos(const std::vector<int>& tt, const std::vector<float>& vv, std::vector<float3>& aos)
{
    aos.clear();
    aos.resize(tt.size());

    for (int i = 0; i < (int) tt.size(); ++i)
    {
        const int t = tt[i];
        const float3 s = {vv[3*t+0], vv[3*t+1], vv[3*t+2]};
        aos[i] = s;
    }
}

#include <map>

void aos2soa(const std::vector<float3>& aos, std::vector<int>& tt, std::vector<float>& vv)
{
    auto comp = [&] (const float3& a, const float3& b) {
        return (a.x < b.x) ||
        (a.x == b.x && a.y < b.y) ||
        (a.x == b.x && a.y == b.y && a.z < b.z);
    };
    
    std::map<float3, int, decltype(comp)> vmap(comp);

    for (const auto& v : aos)
    vmap[v] = -1;
    
    int c = 0;
    for (auto& v : vmap)
    v.second = c++;
    
    vv.resize(3 * vmap.size());
    tt.resize(aos.size());

    c = 0;
    for (auto& v : vmap)
    {
        vv[c++] = v.first.x;
        vv[c++] = v.first.y;
        vv[c++] = v.first.z;
    }

    for (int i = 0; i < (int) aos.size(); ++i)
    {
        const auto v = aos[i];
        const int t = vmap[v];
        tt[i] = t;
    }
}

void subdivide2(std::vector<int>& tt, std::vector<float>& vv)
{    
    vector<float3> aos, aos_fine;

    soa2aos(tt, vv, aos);

    for (uint i = 0; i < aos.size() / 3; ++i)
    {
        const float3 v1 = aos[3*i + 0];
        const float3 v2 = aos[3*i + 1];
        const float3 v3 = aos[3*i + 2];
        
        const float3 h[3] = {
            {(float) 0.5 * (v1.x + v2.x),
             (float) 0.5 * (v1.y + v2.y),
             (float) 0.5 * (v1.z + v2.z)},
            {(float) 0.5 * (v2.x + v3.x),
             (float) 0.5 * (v2.y + v3.y),
             (float) 0.5 * (v2.z + v3.z)},
            {(float) 0.5 * (v3.x + v1.x),
             (float) 0.5 * (v3.y + v1.y),
             (float) 0.5 * (v3.z + v1.z)}};

        const float3 fine[12] = {
            v1, h[0], h[2],
            h[0], v2, h[1],
            h[2], h[1], v3,
            h[0], h[1], h[2]};

        aos_fine.insert(aos_fine.end(), fine, fine + 12);
    }

    aos2soa(aos_fine, tt, vv);
}

void subdivide3(std::vector<int>& tt, std::vector<float>& vv)
{    
    vector<float3> aos, aos_fine;

    soa2aos(tt, vv, aos);

    const float one_third = 1./3.;
    
    for (uint i = 0; i < aos.size() / 3; ++i)
    {
        const float3 v1 = aos[3*i + 0];
        const float3 v2 = aos[3*i + 1];
        const float3 v3 = aos[3*i + 2];

        const float3 h[7] = {
            {one_third * (2 * v1.x + v2.x),
             one_third * (2 * v1.y + v2.y),
             one_third * (2 * v1.z + v2.z)},
            {one_third * (v1.x + 2 * v2.x),
             one_third * (v1.y + 2 * v2.y),
             one_third * (v1.z + 2 * v2.z)},

            {one_third * (2 * v2.x + v3.x),
             one_third * (2 * v2.y + v3.y),
             one_third * (2 * v2.z + v3.z)},
            {one_third * (v2.x + 2 * v3.x),
             one_third * (v2.y + 2 * v3.y),
             one_third * (v2.z + 2 * v3.z)},

            {one_third * (2 * v3.x + v1.x),
             one_third * (2 * v3.y + v1.y),
             one_third * (2 * v3.z + v1.z)},
            {one_third * (v3.x + 2 * v1.x),
             one_third * (v3.y + 2 * v1.y),
             one_third * (v3.z + 2 * v1.z)},
	    
            {one_third * (v1.x + v2.x + v3.x),
             one_third * (v1.y + v2.y + v3.y),
             one_third * (v1.z + v2.z + v3.z)}};
    
            const float3 fine[27] = {
                v1, h[0], h[5], 
                h[5], h[0], h[6],
                h[5], h[6], h[4],
                h[4], h[6], h[3],
                h[4], h[3], v3,
                h[0], h[1], h[6],
                h[6], h[1], h[2],
                h[6], h[2], h[3],
                h[1], v2, h[2]};

        aos_fine.insert(aos_fine.end(), fine, fine + 27);
    }

    aos2soa(aos_fine, tt, vv);
}

void scale_to_usphere(std::vector<float>& vv)
{
    for (uint i = 0; i < vv.size()/3; ++i)
    {
        float *x = vv.data() + 3*i;

        const float sc = 1.f / sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        x[0] *= sc;
        x[1] *= sc;
        x[2] *= sc;
    }
}
