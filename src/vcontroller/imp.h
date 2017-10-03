struct PidVCont {
    
};

void ini(/**/ PidVCont *cont);
void fin(/**/ PidVCont *cont);

void sample(int n, const Particle *pp, /**/ PidVCont *cont);
float3 adjustF(/**/ PidVCont *cont);
