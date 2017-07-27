namespace cnt {
void setup() {
    k_cnt::texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
    k_cnt::texCellsStart.filterMode = cudaFilterModePoint;
    k_cnt::texCellsStart.mipmapFilterMode = cudaFilterModePoint;
    k_cnt::texCellsStart.normalized = 0;

    k_cnt::texCellEntries.channelDesc = cudaCreateChannelDesc<int>();
    k_cnt::texCellEntries.filterMode = cudaFilterModePoint;
    k_cnt::texCellEntries.mipmapFilterMode = cudaFilterModePoint;
    k_cnt::texCellEntries.normalized = 0;
}
}
