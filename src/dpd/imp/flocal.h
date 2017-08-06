void flocal0(float4 *zip0, ushort4 *zip1, int np, int *start, int *count, float seed, float* ff) {
    const int ncells = XS * YS * ZS;

    if( !fdpd_init ) {
        texStartAndCount.channelDesc = cudaCreateChannelDesc<uint2>();
        texStartAndCount.filterMode  = cudaFilterModePoint;
        texStartAndCount.mipmapFilterMode = cudaFilterModePoint;
        texStartAndCount.normalized = 0;

        texParticlesF4.channelDesc = cudaCreateChannelDesc<float4>();
        texParticlesF4.filterMode = cudaFilterModePoint;
        texParticlesF4.mipmapFilterMode = cudaFilterModePoint;
        texParticlesF4.normalized = 0;

        texParticlesH4.channelDesc = cudaCreateChannelDescHalf4();
        texParticlesH4.filterMode = cudaFilterModePoint;
        texParticlesH4.mipmapFilterMode = cudaFilterModePoint;
        texParticlesH4.normalized = 0;

	{
	    is_mps_enabled = false;

	    const char * mps_variables[] = {
		"CRAY_CUDA_MPS",
		"CUDA_MPS",
		"CRAY_CUDA_PROXY",
		"CUDA_PROXY"
	    };

	    for(int i = 0; i < 4; ++i)
		is_mps_enabled |= getenv(mps_variables[i])!= NULL && atoi(getenv(mps_variables[i])) != 0;
	}

        fdpd_init = true;
    }

    static InfoDPD c;

    size_t textureoffset;

    static uint2 *start_and_count;
    static int last_nc;
    if( !start_and_count || last_nc < ncells ) {
        if( start_and_count ) {
            cudaFree( start_and_count );
        }
        cudaMalloc( &start_and_count, sizeof( uint2 )*ncells );
        last_nc = ncells;
    }

    CC( cudaBindTexture( &textureoffset, &texParticlesF4, zip0,  &texParticlesF4.channelDesc, sizeof( float ) * 8 * np ) );
    CC( cudaBindTexture( &textureoffset, &texParticlesH4, zip1, &texParticlesH4.channelDesc, sizeof( ushort4 ) * np ) );
    make_texture2 <<< 64, 512, 0>>>( start_and_count, start, count, ncells );
    CC( cudaBindTexture( &textureoffset, &texStartAndCount, start_and_count, &texStartAndCount.channelDesc, sizeof( uint2 ) * ncells ) );

    c.ncells = make_int3( XS, YS, ZS );
    c.nxyz = XS * YS * ZS;
    c.ff = ff;
    c.seed = seed;

    if (!is_mps_enabled)
	CC( cudaMemcpyToSymbolAsync( info, &c, sizeof( c ), 0, cudaMemcpyHostToDevice) );
    else
	CC( cudaMemcpyToSymbol( info, &c, sizeof( c ), 0, cudaMemcpyHostToDevice ) );

    static int cetriolo = 0;
    cetriolo++;

    int np32 = np;
    if( np32 % 32 ) np32 += 32 - np32 % 32;
    CC( cudaMemsetAsync( ff, 0, sizeof( float )* np32 * 3) );

    if( c.ncells.x % MYCPBX == 0 && c.ncells.y % MYCPBY == 0 && c.ncells.z % MYCPBZ == 0 ) {
        merged<<< dim3( c.ncells.x / MYCPBX, c.ncells.y / MYCPBY, c.ncells.z / MYCPBZ ), dim3( 32, MYWPB ), 0>>> ();
        transpose<<< 28, 1024, 0>>>(np);
    } else {
        fprintf( stderr, "Incompatible grid config\n" );
    }

    CC( cudaPeekAtLastError() );
}

void flocal(float4 *zip0, ushort4 *zip1, int n, int *start, int *count,
	    rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    flocal0(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
}
