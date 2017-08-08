static void tex(float4 *zip0, ushort4 *zip1, int np, int *start, int *count) {
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
    CC( cudaBindTexture( &offset, &texParticlesF4, zip0,  &texParticlesF4.channelDesc, sizeof( float ) * 8 * np ) );
    CC( cudaBindTexture( &offset, &texParticlesH4, zip1, &texParticlesH4.channelDesc, sizeof( ushort4 ) * np ) );
    KL(tex, (64, 512, 0), (start_and_count, start, count, ncells));
    CC( cudaBindTexture( &offset, &texStartAndCount, start_and_count, &texStartAndCount.channelDesc, sizeof( uint2 ) * ncells ) );

}
