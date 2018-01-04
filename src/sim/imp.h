namespace sim {
struct Sim; 
void sim_ini(int argc, char **argv, /**/ Sim **sim);
void sim_gen(Sim *sim);
void sim_strt(Sim *sim);
void sim_fin(Sim *sim);
}
