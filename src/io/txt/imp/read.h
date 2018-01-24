#define VFRMT "%g %g %g"

static void ini(TxtRead **d) {
    UC(emalloc(sizeof(TxtRead), (void**) d));
}

static int get_num_lines(FILE *f) {
    int n;
    char c;
    n = 0;    
    while (EOF != (c = fgetc(f)))
        if (c == '\n') ++n;
    ++n;
    return n;
}

 void txt_read_pp(const char *name, TxtRead **pr) {
    enum {X, Y, Z};
    TxtRead *d;
    FILE *f;
    Particle p;
    int i;
    UC(ini(pr));
    d = *pr;
    d->ff = NULL;

    UC(efopen(name, "r", /**/ &f));
    d->n = get_num_lines(f);
    UC(emalloc(d->n * sizeof(Particle), (void**) &d->pp));
    
    rewind(f);
    i = 0;
    while (6 == fscanf(f, VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z)) {
        d->pp[i++] = p;
    }
    
    UC(efclose(f));
}

void txt_read_pp_ff(const char *name, TxtRead **pr) {
    enum {X, Y, Z};
    TxtRead *d;
    FILE *f;
    Particle p;
    Force fo;
    int i;
    UC(ini(pr));
    d = *pr;

    UC(efopen(name, "r", /**/ &f));
    d->n = get_num_lines(f);
    UC(emalloc(d->n * sizeof(Particle), (void**) &d->pp));
    UC(emalloc(d->n * sizeof(Force),    (void**) &d->ff));
    
    rewind(f);
    i = 0;
    while (9 == fscanf(f, VFRMT " " VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z,
                       fo.f + X, fo.f + Y, fo.f + Z)) {
        d->pp[i] = p;
        d->ff[i] = fo;
        ++i;
    }
    
    UC(efclose(f));
}

void txt_read_fin(TxtRead *d) {
    if (d->pp) UC(efree(d->pp));
    if (d->ff) UC(efree(d->ff));
    UC(efree(d));
}

int txt_read_get_n(const TxtRead *d) {return d->n;}

const Particle* txt_read_get_pp(const TxtRead *d) {return d->pp;}
const Force*    txt_read_get_ff(const TxtRead *d) {return d->ff;}

#undef VFRMT
