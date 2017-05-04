namespace mesh
{
    int in_mesh_1p(const float *r, const float *vv, const int *tt, const int nt)
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

    void in_mesh_hst(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        for (int i = 0; i < n; ++i)
        inout[i] = in_mesh_1p(rr + 3*i, vv, tt, nt);
    }

    void in_mesh_dev(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        k_mesh::in_mesh_k <<< k_cnf(n) >>>(rr, n, vv, tt, nt, /**/ inout);
    }
}
