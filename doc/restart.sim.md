# How sim plays with restart?

    main:
	  ini
	  sim
	  fin

    sim:
	  if Generate:
	    run_ini
	    run_eq
	    freeze
	    run_sim
	    run_fin
	  else
	    read
	    run_ini
	    run_sim
	    run_fin
