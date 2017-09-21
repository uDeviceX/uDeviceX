struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
};

typedef Sarray<Particle*, 26> Particlep26;
typedef Sarray<Force*, 26>    Forcep26;
