namespace sub {

int genColor(/*o*/ Particle *dev, int *cc, /*w*/ Particle *hst);
int genGrey(/*o*/ Particle *dev, /*w*/ Particle *hst);

void ii_gen(const int n, int *ii_dev, int *ii_hst);

void tags0_gen(const int n, int *ii_dev, int *ii_hst);

int strt(const int id, Particle *dev, /*w*/ Particle *hst);
int strt_ii(const char *subext, const int id, int *dev, /*w*/ int *hst);

void strt_dump(const int id, const int n, const Particle *dev, Particle *hst);
void strt_dump_ii(const char *subext, const int id, const int n, const int *dev, int *hst);

void zip(const Particle *pp, const int n, /**/ float4 *zip0, ushort4 * zip1);

} // sub
