# -*- coding: utf-8 -*-
# #include "navier-stokes/centered.h"
# #include "./ibm.h" 
# #include "view.h"
# 
# #define L0 20.
# #define D 0.5
# #define LEVEL 9
# 
# int Re;
# int maxlevel = 9;
# double U0 =  1.;             // inlet velocity
# double t_end = 110;
# coord ci = {L0/4, L0/2};     // initial coordinates of cylinder
# coord vc = {0, 0};           // velocity of cylinder
# 
# scalar vof[];
# face vector sf[];
# face vector muv[];
# 
# u.n[left] = dirichlet (U0);
# u.t[left] = dirichlet (0);
# p[left]   = neumann (0);
# pf[left]  = neumann (0);
# 
# u.n[right] = neumann (0);
# u.t[right] = neumann (0);
# p[right]   = dirichlet (0);
# pf[right]  = dirichlet (0);
# 
# u.n[top] = neumann (0);
# 
# u.n[bottom] = neumann (0);
# 
# int main() {
#   size(L0);
#   init_grid (1 << (LEVEL - 2));
#   mu = muv;
#   TOLERANCE = 1.e-6; 
#   CFL = 0.8;
# 
# /*
#   Re = 40;
#   run();
# 
#   Re = 80;
#   run();
# 
#   Re = 120;
#   run();
# */
#   Re = 20;
#   run();
# 
#   Re = 180;
#   run();
# }
# 
# event init (t = 0) {
#   solid (vof, sf, - sq(x - ci.x - vc.x) - sq(y - ci.y - vc.y) + sq(D/2));
#   refine (vof[] < 1 && vof[] > 0 && level < LEVEL);
# }
# 
# event moving_cylinder (i++) {
#   solid (vof, sf, - sq(x - ci.x - vc.x) - sq(y - ci.y - vc.y) + sq(D/2));
# }
# 
# event properties (i++) {
#   foreach_face()
#     muv.x[] = fm.x[]*(U0)*(D)/(Re);
#    boundary ((scalar *) {muv});
# }
# 
# event logfile (i++) {
#   coord F = ibm_force();
#   double CD = (F.x)/(0.5*sq(U0)*(D));
#   double CL = (F.y)/(0.5*sq(U0)*(D));
#   fprintf (stderr, "%d %g %d %d %d %d %d %g %g\n",
#           i, t, Re, mgp.i, mgp.nrelax, mgu.i, mgu.nrelax, CD, CL);
# }
# 
# event snapshot (t = t_end) {
#   scalar omega[];
#   vorticity (u, omega);
# 
#   char name[80];
#   view (fov = 2, tx = -0.25, ty = -0.465,
#         width = 3000, height = 1500); 
#   sprintf (name, "%g-pressure-%d.png", t, Re);
#   clear();
#   draw_vof ("vof", "sf", lw = 5, lc = {0,0,0});
#   squares ("p", min = -0.5, max = 0.5, map = blue_white_red);
#   save (name);
# 
#   sprintf (name, "%g-vort-%d.png", t, Re);
#   clear();
#   squares ("omega", min = -3, max = 3, map = blue_white_red);
#   draw_vof ("vof", "sf", lw = 5, lc = {0,0,0});
#   save (name);
# 
#   sprintf (name, "%g-pressureiso-%d.png", t, Re);
#   clear();
#   draw_vof ("vof", "sf", lw = 5, lc = {0,0,0});
#   isoline ("p", n = 20, min = -0.5, max = 0.5, lc = {0,0,0});
#   squares ("p", min = -0.5, max = 0.5, map = blue_white_red);
#   save (name);
# 
#   sprintf (name, "%g-vortiso-%d.png", t, Re);
#   clear();
#   isoline ("omega", n = 15, min = -3, max = 3, lc = {0,0,0});
#   squares ("omega", min = -3, max = 3, map = blue_white_red);
#   draw_vof ("vof", "sf", lw = 5,  lc = {0,0,0});
#   save (name);
# }
# 
# event adapt (i++) {
#   adapt_wavelet ({vof,u}, (double[]){1.e-2,3e-3,3e-3},
# 		 maxlevel = LEVEL, minlevel = 2);
# }
# 
# 
# event fields (t = t_end-38.5; t += 0.077) {
#   char name[80];
#   sprintf(name, "%d-data-%g", Re, t);
#   FILE * file = fopen(name, "w");
#   output_field (list = {u.x, u.y, p}, fp = file, n = 255, box={{4, 7}, {10, 13}});
#   fclose(file);
# }
# 
# /*
# event movie (t = t_end-7.7; t += 0.077)
# {
#     char name[80];
#     sprintf (name, "%d-xmovie-%g.png", Re, t);
#     view (fov = 2, tx = -0.3, ty = -0.5,
#             width = 1500, height = 750);
#     clear();
#     squares ("u.x", min=-1, max=1, map = blue_white_red);
#     draw_vof ("vof", "sf", lw=5, lc = {0,0,0});
#     save (name);
# 
#     sprintf (name, "%d-ymovie-%g.png", Re, t);
#     clear();
#     squares ("u.y", min=-1, max=1, map = blue_white_red);
#     draw_vof ("vof", "sf", lw=5, lc = {0,0,0});
#     save (name);
# 
#     sprintf (name, "%d-pmovie-%g.png", Re, t);
#     clear();
#     squares ("p", min=-0.5, max=0.5, map = blue_white_red);
#     draw_vof ("vof", "sf", lw=5, lc = {0,0,0});
#     save (name);
# }
# */
# 
# event stop (t = t_end) {
#   static FILE * fp = fopen("perf", "w");
#   timing s = timer_timing (perf.gt, iter, perf.tnc, NULL);
#   fprintf (fp, "%d\t%g\t%d\t%g\n", Re, s.real, i, s.speed);
#   fflush (fp);
#   return 1;
# }
# 
