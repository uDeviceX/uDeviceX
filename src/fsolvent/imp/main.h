static void zip(const int n, const Particle *pp, /**/ float4 *zip0, ushort4 * zip1) {
    assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
    KL(dev::zip, (k_cnf(n)), (n, (float*)pp, zip0, zip1));
}


void prepare(const Cloud *c, /**/ BulkData *b) {
    
}
