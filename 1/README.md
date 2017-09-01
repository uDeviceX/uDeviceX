# status

`ssh daint sh /scratch/snx1600/lisergey/check`

# FIX/gold/wall0

	04 void step0(float driving_force0, bool wall0, int it) {
	05     if (solids0) distr_solid();
	06     if (rbcs)    distr_rbc();
	07     forces(wall0);
	08     dump_diag0(it);
	09     if (wall0 || solids0) dump_diag_after(it);
	10     body_force(driving_force0);
	11     update_solvent();
	12     if (solids0) update_solid();
	13     if (rbcs)    update_rbc();
	14     O;
	15     if (wall0) bounce();
	16     O;
	17     if (sbounce_back && solids0) bounce_solid(it);
	18     O;
	19 }


	:  1.5e+03  1.2e+00 [ 1.8e+05  2.2e+01 -1.3e+01]  1.0e+01
	../src/sim/_dang/step.h:14: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:14: (sub::check_vv): NAN_VAL ov
	../src/sim/_dang/step.h:16: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:16: (sub::check_vv): NAN_VAL ov
	../src/sim/_dang/step.h:18: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:18: (sub::check_vv): NAN_VAL ov
	../src/odstr/halo/_dang/check.h:22: void dev::check(const float *): block: [20,0,0], thread: [106,0,0] Assertion `0` failed.
	wild particle: [nan nan nan]

# FIX/gold/center0

	09     if (wall0 || solids0) dump_diag_after(it);
	10     body_force(driving_force0);
	11     O;
	12     update_solvent();
	13     O;
	14     if (solids0) update_solid();
	15     if (rbcs)    update_rbc();
	16     O;
	17     if (wall0) bounce();
	18     O;
	19     if (sbounce_back && solids0) bounce_solid(it);
	20     O;


	:  7.0e+02  2.2e+00 [ 1.9e+05 -7.9e-02  1.2e+01]  2.0e+01
	../src/sim/_dang/step.h:13: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:13: (sub::check_vv): NAN_VAL ov
	../src/sim/_dang/step.h:16: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:16: (sub::check_vv): NAN_VAL ov
	../src/sim/_dang/step.h:18: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:18: (sub::check_vv): NAN_VAL ov
	../src/sim/_dang/step.h:20: (sub::check_pos_pu): NAN_VAL op
	../src/sim/_dang/step.h:20: (sub::check_vv): NAN_VAL ov
