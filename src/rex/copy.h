namespace rex {
void copy_pack(x::TicketPinned t) {
    if (t.tstarts[26]) CC(cudaMemcpyAsync(host_packbuf, packbuf, sizeof(Particle) * t.tstarts[26], H2H));
}
}
