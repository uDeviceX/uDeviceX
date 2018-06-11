struct Disp;
struct Com;

void disp_ini(int n, Disp **disp);
void disp_fin(Disp *d);

void disp_add(int n, const Com *cc0, const Com *cc1, int L[3], Disp *d);
void disp_reduce(const Disp *d, float tot[3]);
