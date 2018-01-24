struct Particle;
struct Force;

void punto_write_pp(long n, const Particle*, const char *name);
void punto_write_pp_ff(long n, const Particle*, const Force*, const char *name);

struct PuntoRead;

void punto_read_pp(const char *name, PuntoRead **);
void punto_read_pp_ff(const char *name, PuntoRead **);
void punto_read_fin(PuntoRead*);

int punto_read_get_n(const PuntoRead *);
const Particle* punto_read_get_pp(const PuntoRead *);
const Force*    punto_read_get_ff(const PuntoRead *);
