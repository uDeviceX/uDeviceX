namespace collision
{
int inside_1p(const float *r, const float *vv, const int *tt, const int nt);
void inside_hst(const Particle *pp, const int n, const Mesh m, /**/ int *inout);

/* tags: -1 if outside, i if in ith solid */
void inside_hst(const Particle *pp, const int n, const Mesh m, const Particle *i_pp, const int ns, /**/ int *tags);
void inside_dev(const Particle *pp, const int n, const Mesh m, const Particle *i_pp, const int ns, /**/ int *tags);
    
void get_bbox(const float *rr, const int n, /**/ float *bbox);

void get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float *bboxes);
void get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
}
