/*
 * example2.c: oscillating mesh with FORMS control panel
 *
 * This example module is distributed with the geomview manual.
 * If you are not reading this in the manual, see the "External
 * Modules" chapter of the manual for an explanation.
 *
 * This module creates an oscillating mesh and has a FORMS control
 * panel that lets you change the speed of the oscillation with a
 * slider.
 */

#include <math.h>
#include <stdio.h>
#include <sys/time.h>           /* for struct timeval below */

#include "forms.h"              /* for FORMS library */

FL_FORM *OurForm;
FL_OBJECT *VelocitySlider;
float dt;

/* F is the function that we plot
 */
float F(x,y,t)
     float x,y,t;
{
  float r = sqrt(x*x+y*y);
  return(sin(r + t)*sqrt(r));
}

/* SetVelocity is the slider callback procedure; FORMS calls this
 * when the user moves the slider bar.
 */
void SetVelocity(FL_OBJECT *obj, long val)
{
  dt = fl_get_slider_value(VelocitySlider);
}

/* Quit is the "Quit" button callback procedure; FORMS calls this
 * when the user clicks the "Quit" button.
 */
void Quit(FL_OBJECT *obj, long val)
{
  exit(0);
}

/* create_form_OurForm() creates the FORMS panel by calling a bunch of
 * procedures in the FORMS library.  This code was generated
 * automatically by the FORMS designer program; normally this code
 * would be in a separate file which you would not edit by hand.  For
 * simplicity of this example, however, we include this code here.
 */
create_form_OurForm()
{
  FL_OBJECT *obj;
  FL_FORM *form;
  OurForm = form = fl_bgn_form(FL_NO_BOX,380.0,120.0);
  obj = fl_add_box(FL_UP_BOX,0.0,0.0,380.0,120.0,"");
  VelocitySlider = obj = fl_add_valslider(FL_HOR_SLIDER,20.0,30.0,
                                          340.0,40.0,"Velocity");
    fl_set_object_lsize(obj,FL_LARGE_FONT);
    fl_set_object_align(obj,FL_ALIGN_TOP);
    fl_set_call_back(obj,SetVelocity,0);
  obj = fl_add_button(FL_NORMAL_BUTTON,290.0,75.0,70.0,35.0,"Quit");
    fl_set_object_lsize(obj,FL_LARGE_FONT);
    fl_set_call_back(obj,Quit,0);
  fl_end_form();
}

main(argc, argv)        
     char **argv;
{
  int xdim, ydim;
  float xmin, xmax, ymin, ymax, dx, dy, t;
  int fdmask;
  static struct timeval timeout = {0, 200000};

  xmin = ymin = -5;             /* Set x and y            */
  xmax = ymax = 5;              /*    plot ranges         */
  xdim = ydim = 24;             /* Set x and y resolution */
  dt = 0.1;                     /* Time increment is 0.1  */

  /* Forms panel setup.
   */
  foreground();
  create_form_OurForm();
  fl_set_slider_bounds(VelocitySlider, 0.0, 1.0);
  fl_set_slider_value(VelocitySlider, dt);
  fl_show_form(OurForm, FL_PLACE_SIZE, TRUE, "Example 2");


  /* Geomview setup.
   */
  printf("(geometry example { : foo })\n");
  fflush(stdout);

  /* Loop until killed.
   */
  for (t=0; ; t+=dt) {
    fdmask = (1 << fileno(stdin)) | (1 << qgetfd());
    select(qgetfd()+1, &fdmask, NULL, NULL, &timeout);
    fl_check_forms();
    UpdateMesh(xmin, xmax, ymin, ymax, xdim, ydim, t);
  }
}

/* UpdateMesh sends one mesh iteration to geomview
 */
UpdateMesh(xmin, xmax, ymin, ymax, xdim, ydim, t)
     float xmin, xmax, ymin, ymax, t;
     int xdim, ydim;
{
  int i,j;
  float x,y, dx,dy;

  dx = (xmax-xmin)/(xdim-1);
  dy = (ymax-ymin)/(ydim-1);

  printf("(read geometry { define foo \n");
  printf("MESH\n");
  printf("%1d %1d\n", xdim, ydim);
  for (j=0, y = ymin; j<ydim; ++j, y += dy) {
    for (i=0, x = xmin; i<xdim; ++i, x += dx) {
      printf("%f %f %f\t", x, y, F(x,y,t));
    }
    printf("\n");
  }
  printf("})\n");
  fflush(stdout);
}
