namespace fsi {
void halo(ParticlesWrap halos[26], Pap26 PP, Fop26 FF, int nn[26]);
void ini();
void fin();
void bind(SolventWrap wrap);
void bulk(std::vector<ParticlesWrap> wr);
}
