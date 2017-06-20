# How sim plays with restart?

    main:
	  ini
	  sim
	  fin

    sim:
	  if Generate:
	    run_eq
	    freeze
	    run_sim
	    run_fin
	  else
	    read
	    run_sim
	    run_fin
