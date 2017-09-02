static void tex_cells(int *start, int *count) {
    const int ncells = XS * YS * ZS;
    static uint2 *start_and_count;
    static int last_nc;
    if( !start_and_count || last_nc < ncells ) {
        if( start_and_count ) {
            cudaFree( start_and_count );
        }
        cudaMalloc( &start_and_count, sizeof( uint2 )*ncells );
        last_nc = ncells;
    }

    size_t offset;
    KL(tex, (64, 512), (start_and_count, start, count, ncells));
    CC(cudaBindTexture( &offset, &texStartAndCount, start_and_count, &texStartAndCount.channelDesc, sizeof( uint2 ) * ncells));

}
