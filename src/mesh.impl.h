namespace mesh
{
    int inside_1p(const float *r, const float *vv, const int *tt, const int nt)
    {
        int c = 0;

        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int *t = tt + 3 * i;

            if (k_mesh::in_tetrahedron(r, vv + 3*t[0], vv + 3*t[1], vv + 3*t[2], origin)) ++c;
        }
        
        return (c+1)%2;
    }

    void inside_hst(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        for (int i = 0; i < n; ++i)
        inout[i] = inside_1p(pp[i].r, vv, tt, nt);
    }

    void inside_dev(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        k_mesh::inside <<< k_cnf(n) >>>(pp, n, vv, tt, nt, /**/ inout);
    }
}
