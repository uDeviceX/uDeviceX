struct Particle;
struct Force;

void txt_write_pp(long n, const Particle*, const char *name);
void txt_write_pp_ff(long n, const Particle*, const Force*, const char *name);

struct TxtRead;

void txt_read_pp(const char *name, TxtRead **);
void txt_read_pp_ff(const char *name, TxtRead **);
void txt_read_fin(TxtRead*);

int txt_read_get_n(const TxtRead *);
const Particle* txt_read_get_pp(const TxtRead *);
const Force*    txt_read_get_ff(const TxtRead *);
