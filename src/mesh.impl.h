namespace mesh
{
    enum {X, Y, Z};
    
    static int inside_1p(const float *r, const float *vv, const int *tt, const int nt)
    {
        int c = 0;

        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int *t = tt + 3 * i;

            if (k_mesh::in_tetrahedron(r, vv + 3*t[0], vv + 3*t[1], vv + 3*t[2], origin)) ++c;
        }
        
        return c%2;
    }

    void inside_hst(const Particle *pp, const int n, const Mesh m, /**/ int *inout)
    {
        for (int i = 0; i < n; ++i)
        inout[i] = inside_1p(pp[i].r, m.vv, m.tt, m.nt);
    }

    void inside_dev(const Particle *pp, const int n, const Mesh m, /**/ int *inout)
    {
        k_mesh::inside <<< k_cnf(n) >>>(pp, n, m.vv, m.tt, m.nt, /**/ inout);
    }

    /* bbox: minx, maxx, miny, maxy, minz, maxz */
    void bbox(const Particle *pp, const int n, /**/ float *bbox)
    {
        if (n == 0) return;

        const float *r = pp[0].r;
        
        bbox[0] = bbox[1] = r[0];
        bbox[2] = bbox[3] = r[1];
        bbox[4] = bbox[5] = r[2];

        auto higher = [](float a, float b) {return a > b ? a : b;};
        auto lower  = [](float a, float b) {return a > b ? b : a;};

        for (int i = 1; i < n; ++i)
        {
            r = pp[i].r;
            for (int d = 0; d < 3; ++d)
            {
                bbox[2*d + 0] =  lower(bbox[2*d + 0], r[d]);
                bbox[2*d + 1] = higher(bbox[2*d + 1], r[d]); 
            }
        }
    }

    void bbox(const float *rr, const int n, /**/ float *bbox)
    {
        if (n == 0) return;

        const float *r = rr;
        
        bbox[0] = bbox[1] = r[0];
        bbox[2] = bbox[3] = r[1];
        bbox[4] = bbox[5] = r[2];

        auto higher = [](float a, float b) {return a > b ? a : b;};
        auto lower  = [](float a, float b) {return a > b ? b : a;};

        for (int i = 1; i < n; ++i)
        {
            r = rr + 3 * i;;
            for (int d = 0; d < 3; ++d)
            {
                bbox[2*d + 0] =  lower(bbox[2*d + 0], r[d]);
                bbox[2*d + 1] = higher(bbox[2*d + 1], r[d]); 
            }
        }
    }

    void bboxes_hst(const Particle *pp, const int np, const int ns, /**/ float *bboxes)
    {
        for (int i = 0; i < ns; ++i)
        bbox(pp, np, /**/ bboxes + 6 * i);
    }

    void bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float *bboxes)
    {
        minmax(pp, nps, ns, /**/ bboxes);
    }
}
