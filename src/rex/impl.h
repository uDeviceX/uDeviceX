namespace rex {

void recv_copy_bags() {
    float *recvbags[26];
    for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i]->result->DP;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::recvbags, recvbags, sizeof(recvbags), 0, H2D));
}

}
