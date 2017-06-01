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

void scale(std::vector<float>& vv, const float sc)
{
    for (uint i = 0; i < vv.size()/3; ++i)
    {
        float *x = vv.data() + 3*i;
        x[0] *= sc;
        x[1] *= sc;
        x[2] *= sc;
    }
}

static int min2(int a, int b) {return a < b ? a : b;}
static int max2(int a, int b) {return a < b ? b : a;}

struct Edge
{
    Edge() : v {-1, -1} {};
    Edge(int v1, int v2) : v {min2(v1, v2), max2(v1, v2)} {}

    bool operator< (const Edge o) const {return (v[0] < o.v[0]) || (v[0] == o.v[0] && v[1] < o.v[1]);}
    
    int v[2];
};

static void swap3(int *i) // 1, 2, 3 -> 3, 1, 2
{
    const int tmp = i[2];
    i[2] = i[1]; i[1] = i[0];
    i[0] = tmp;
}

static bool ordered(int *t, const Edge e)
{
    return (e.v[0] == t[0] && e.v[1] == t[1]) || (e.v[0] == t[1] && e.v[1] == t[0]);
}

static bool no_need_flip(int a, int b, int c, int d, const float *vv)
{
#define load_v(t) {vv[3*t+0], vv[3*t+1]}

    const float A[2] = load_v(a);
    const float B[2] = load_v(b);
    const float C[2] = load_v(c);
    const float D[2] = load_v(d);

    enum {X, Y, Z};
    
    const float M[3][3] = {
        { A[X] - D[X], A[Y] - D[Y], (A[X]*A[X] - D[X]*D[X]) + (A[Y]*A[Y] - D[Y]*D[Y]) },
        { B[X] - D[X], B[Y] - D[Y], (B[X]*B[X] - D[X]*D[X]) + (B[Y]*B[Y] - D[Y]*D[Y]) },
        { C[X] - D[X], C[Y] - D[Y], (C[X]*C[X] - D[X]*D[X]) + (C[Y]*C[Y] - D[Y]*D[Y]) } };

    const float detM =
        + M[X][X] * (M[Y][Y] * M[Z][Z] - M[Y][Z] * M[Z][Y])
        - M[X][Y] * (M[Y][X] * M[Z][Z] - M[Y][Z] * M[Z][X])
        + M[X][Z] * (M[Y][X] * M[Z][Y] - M[Y][Y] * M[Z][X]);
    
#undef load_v

    return detM < 0;
}
    

#include <map>

int flip_edges(std::vector<int>& tt, const std::vector<float>& vv)
{
    std::map <Edge, std::vector<int> > adj;

    int c = 0;
    
    for (int it = 0; it < (int) tt.size()/3; ++it)
    {
        const int t1 = tt[3*it + 0];
        const int t2 = tt[3*it + 1];
        const int t3 = tt[3*it + 2];

        adj[Edge(t1, t2)].push_back(it);
        adj[Edge(t1, t3)].push_back(it);
        adj[Edge(t3, t2)].push_back(it);
    }
           
    for (auto e = adj.begin(); e != adj.end(); /**/)
    {
        if (e->second.size() != 2)
        e = adj.erase(e);
        else
        e++;
    }

    for (auto ite = adj.begin(); ite != adj.end(); /**/)
    {
        const int it1 = ite->second[0];
        const int it2 = ite->second[1];
        
#define load_t(it) {tt[3*it + 0], tt[3*it + 1], tt[3*it + 2]}

        int t1[3] = load_t(it1);
        int t2[3] = load_t(it2);

#undef load_t
        
        while (!ordered(t1, ite->first)) swap3(t1);
        while (!ordered(t2, ite->first)) swap3(t2);
        
        if (no_need_flip(t1[0], t1[2], t1[1], t2[2], vv.data()))
        {
            ite ++;
            continue;
        }
        
        const int t1n[3] = {t1[2], t1[0], t2[2]};
        const int t2n[3] = {t1[2], t2[2], t2[0]};

#define write_nt(it, t) do { tt[3*it + 0] = t[0]; tt[3*it + 1] = t[1]; tt[3*it + 2] = t[2]; } while (0)

        write_nt(it1, t1n);
        write_nt(it2, t2n);

#undef write_nt
        
        {
            const Edge sharede(t1[0], t1[1]);
            ite = adj.erase(ite);
            adj[sharede].push_back(it1);
            adj[sharede].push_back(it2);
        }

        {
            const Edge e1to2(t1[1], t1[2]);

            if (adj.find(e1to2) != adj.end())
            {
                auto& ts = adj[e1to2];

                if      (ts[0] == it1) ts[0] = it2;
                else if (ts[1] == it1) ts[1] = it2;

                if (ts[0] > ts[1]) {int tmp = ts[0]; ts[0] = ts[1]; ts[1] = tmp;}
            }
        }

        {
            const Edge e2to1(t2[1], t2[2]);

            if (adj.find(e2to1) != adj.end())
            {
                auto& ts = adj[e2to1];

                if      (ts[0] == it2) ts[0] = it1;
                else if (ts[1] == it2) ts[1] = it1;

                if (ts[0] > ts[1]) {int tmp = ts[0]; ts[0] = ts[1]; ts[1] = tmp;}
            }
        }

        ++c;
    }
    return c;
}
