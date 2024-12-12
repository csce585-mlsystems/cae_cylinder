extern scalar vof;
extern face vector sf;


#define distance(a,b) sqrt(sq(a) + sq(b))

void bilinear_interpolation (Point point, vector uv, coord pc, coord * uc)
{
    coord uci;
    double xc = pc.x;
    double yc = pc.y;

    double xx = x - xc;
    double yy = y - yc;

    double x2 = x - sign(xx)*Delta;
    double y2 = y - sign(yy)*Delta;

    foreach_dimension() {
        uci.x = (x2 - xc)*(y2 - yc) * uv.x[] +
                (xc - x) * (y2 - yc) * uv.x[-sign(xx)] +
        	    (x2 - xc) * (yc - y) * uv.x[0,-sign(yy)] +
	            (xc - x) * (yc - y) * uv.x[-sign(xx),-sign(yy)];
        uci.x /=  ((x2 - x)*(y2 - y));
    }

    *uc = uci;
}

double scalar_bilinear_interpolation (Point point, scalar p, coord pc) 
{
    double pci;
    double xc = pc.x;
    double yc = pc.y;

    double xx = x - xc;
    double yy = y - yc;

    double x2 = x - sign(xx)*Delta;
    double y2 = y - sign(yy)*Delta;

    pci = (x2 - xc)*(y2 - yc) * p[] +
          (xc - x) * (y2 - yc) * p[-sign(xx)] +
          (x2 - xc) * (yc - y) * p[0,-sign(yy)] +
	      (xc - x) * (yc - y) * p[-sign(xx),-sign(yy)];
    pci /=  ((x2 - x)*(y2 - y));

  return pci;
}

double phi_func (double x, double h) 
{
    double r = x / h;
    double phi;

    if (fabs(r) <= 1)
      phi = (1./8.) * (3 - (2 * fabs(r)) + sqrt (1 + (4 * fabs(r)) - (4 * sq(r))));
    else if (fabs(r) > 1 && fabs(r) <= 2)
      phi = (1./8.) * (5 - (2 * fabs(r)) - sqrt (-7 + (12 * fabs(r)) - ( 4 * sq(r))));
    else
      phi = 0;
    return phi;
}

double phi_smooth (double x, double h)
{
    double r = x / h;
    double phi;

    if (fabs(r) <= 0.5)
        phi = 3./8. + M_PI/32. - sq(r)/4.;
    else if (fabs(r) > 0.5 && fabs(r) <= 1.5)
        phi = 1./4. + (1. - fabs(r)) / (8.) * sqrt (-2. + 8. * fabs(r) - 4*sq(r)) - 1./8. * asin(sqrt(2.) * (fabs(r) - 1.));
    else if (fabs(r) > 1.5 && fabs(r) <= 2.5)
        phi = 17./16. - M_PI/64. - (3*fabs(r))/4. + sq(r)/8. + (fabs(r) - 2)/16. * sqrt (-14. + 16*fabs(r) - 4*sq(r)) + 1./16. * asin(sqrt(2.) * (fabs(r) - 2));
    else
        phi = 0.;

    return phi;
}

double phi_three (double x, double h)
{
    double r = x / h;
    double phi;

    if (fabs(r) <= 0.5)
        phi = 1./3. * (1 + sqrt(1 - 3 * sq(r)));
    else if (fabs(r) <= 1.5 && fabs(r) > 0.5)
        phi = 1./6. * (5 - 3 * fabs(r) - sqrt (1 - 3 * sq(1 - fabs(r))));
    else
        phi = 0.;

    return phi;
}

double delta_func (double x, double y, double xc, double yc, double Delta, double z = 0, double zc = 0)
{
    double phi_x = phi_func (x - xc, Delta);
    double phi_y = phi_func (y - yc, Delta);
#if dimension == 3
    double phi_z = phi_func (z - zc, Delta);
    return (phi_x * phi_y * phi_z) / pow(Delta, 3);
#else
    return (phi_x * phi_y) / sq(Delta);
#endif
}

bool empty_neighbor (Point point, coord * pc, scalar vof)
{
    coord pc_temp;
    double temp_vof = vof[];
    double xc = x;
    double yc = y;
    double max_d = 1e6;
    int neighbor = 0;

    foreach_neighbor(1)
        if (vof[] == 0 && temp_vof == 1 && (distance(x - xc, y - yc) < max_d)) {
            pc_temp.x = (xc + x) / 2.;
            pc_temp.y = (yc + y) / 2.;
            max_d = distance(x - xc, y - yc);
            neighbor = 1;
            *pc = pc_temp;
        }
   return neighbor;
}

double marker_point (Point point, scalar vof, coord * markerPoint)
{
#if dimension == 3
    coord cellCenter = {x, y, z};
#else
    coord cellCenter = {x, y};
#endif
    coord n = interface_normal (point, vof);
    double alpha = plane_alpha (vof[], n);
    double area = plane_area_center (n, alpha, markerPoint);

    foreach_dimension()
        markerPoint->x = cellCenter.x + markerPoint->x*Delta;
    return area;
}

#define quadratic(x,a1,a2,a3) \
  (((a1)*((x) - 1.) + (a3)*((x) + 1.))*(x)/2. - (a2)*((x) - 1.)*((x) + 1.))

foreach_dimension()
static inline double dirichlet_gradient_x (Point point, scalar s, scalar cs,
					   coord n, coord p, coord bc, double * coef)
{
    
    double d[2] = {0,0}, v[2] = {nodata,nodata};
    bool defined = true;
    for (int l = 0; l <= 1; l++) {
        int i = (l + 1)*sign(n.x);
        d[l] = (i - p.x)/n.x;
        double y1 = p.y + d[l]*n.y;
        int j = y1 > 0.5 ? 1 : y1 < -0.5 ? -1 : 0;
        y1 -= j;
        if (cs[i,j-1] < 0.5 && cs[i,j] < 0.5 && cs[i,j+1] < 0.5)
            v[l] = quadratic (y1, (s[i,j-1]), (s[i,j]), (s[i,j+1]));
    }
    if (v[0] == nodata) {
        d[0] = max(1e-3, fabs(p.x/n.x));
        *coef = - 1./(d[0]*Delta);
        return bc.x/(d[0]*Delta);
    }
    *coef = 0.;
    double gradient = 0;
    if (v[1] != nodata) // third-order gradient
        gradient = (d[1]*(bc.x - v[0])/d[0] - d[0]*(bc.x - v[1])/d[1])/((d[1] - d[0])*Delta);
    else
        gradient = (bc.x - v[0])/(d[0]*Delta); // second-order gradient
    return gradient;
}


double dirichlet_gradient (Point point, scalar s, scalar cs,
			   coord n, coord p, coord bc, double * coef)
{
#if dimension == 2
  foreach_dimension()
    if (fabs(n.x) >= fabs(n.y))
      return dirichlet_gradient_x (point, s, cs, n, p, bc, coef);
#else // dimension == 3
  if (fabs(n.x) >= fabs(n.y)) {
    if (fabs(n.x) >= fabs(n.z))
      return dirichlet_gradient_x (point, s, cs, n, p, bc, coef);
  }
  else if (fabs(n.y) >= fabs(n.z))
    return dirichlet_gradient_y (point, s, cs, n, p, bc, coef);
  return dirichlet_gradient_z (point, s, cs, n, p, bc, coef);
#endif // dimension == 3
  return nodata;
}

double embed_geometry (Point point, coord * b, coord * n)
{
  *n = facet_normal (point, vof, sf);
  double alpha = plane_alpha (vof[], *n);
  double area = plane_area_center (*n, alpha, b);
  normalize (n);
  return area;
}

double embed_interpolate (Point point, scalar s, coord p)
{
  assert (dimension == 2);
  int i = sign(p.x), j = sign(p.y);
  if (vof[i] < 0.5 && vof[0,j] < 0.5 && vof[i,j] < 0.5)
    // bilinear interpolation when all neighbors are defined
    return ((s[]*(1. - fabs(p.x)) + s[i]*fabs(p.x))*(1. - fabs(p.y)) + 
	    (s[0,j]*(1. - fabs(p.x)) + s[i,j]*fabs(p.x))*fabs(p.y));
  else {
    // linear interpolation with gradients biased toward the
    // cells which are defined
    double val = s[];
    foreach_dimension() {
      int i = sign(p.x);
      if (vof[i])
	val += fabs(p.x)*(s[i] - s[]);
      else if (vof[-i])
	val += fabs(p.x)*(s[] - s[-i]);
    }
    return val;
  }
}


static inline
coord embed_gradient (Point point, vector u, coord p, coord n, coord bc)
{
  coord markerCoord, boundaryCondition, dudn;

  marker_point(point, vof, &markerCoord);
  bilinear_interpolation(point, u, markerCoord, &boundaryCondition);

  foreach_dimension() {
    bool dirichlet = true;
    if (dirichlet) {
      double val;
      dudn.x = dirichlet_gradient (point, u.x, vof, n, p, boundaryCondition, &val);
      dudn.x += u.x[]*val;
    }
    else // Neumann
      dudn.x = bc.x;
    if (dudn.x == nodata)
      dudn.x = 0.;
  }

    // fprintf (stderr, "|| x=%g y=%g mC.x=%g mC.y=%g bC.x=%g bC.y=%g dudn.x=%g dudn.y=%g\n",
    //         x, y, markerCoord.x, markerCoord.y, boundaryCondition.x, boundaryCondition.y, dudn.x, dudn.y);

  return dudn;
}


double ibm_dirichlet_gradient (Point point, scalar uc, coord n, coord markerCoord, double boundaryCondition)
{
    coord cellCenter = {x, y}, distance, normalPoint;

    foreach_dimension() 
        distance.x = (Delta * n.x) - (markerCoord.x - cellCenter.x);
   
    if (fabs(n.x) >= fabs(n.y)) {
        normalPoint.x = cellCenter.x + sign(n.x)*Delta;
        normalPoint.y = markerCoord.y + distance.y;
    }
    else if (fabs(n.x) < fabs(n.y)) {
        normalPoint.x = markerCoord.x + distance.x;
        normalPoint.y = cellCenter.y + sign(n.y)*Delta;
    }
    else 
        return 0.;

    double normalVelocity = scalar_bilinear_interpolation(point, uc, normalPoint);
    double totalDistance = distance(markerCoord.x - normalPoint.x, markerCoord.y - normalPoint.y);


    return (normalVelocity - boundaryCondition) / totalDistance;
}


coord ibm_gradient (Point point, vector u, coord markerCoord, coord n)
{
    coord dudn, boundaryCondition;

    marker_point(point, vof, &markerCoord);
    bilinear_interpolation(point, u, markerCoord, &boundaryCondition);

    foreach_dimension()
        dudn.x = -ibm_dirichlet_gradient (point, u.x, n, markerCoord, boundaryCondition.x);
/*
    fprintf (stderr, "|| x=%g y=%g mC.x=%g mC.y=%g bC.x=%g bC.y=%g dudn.x=%g dudn.y=%g\n",
             x, y, markerCoord.x, markerCoord.y, boundaryCondition.x, boundaryCondition.y, dudn.x, dudn.y);
*/            
    return dudn;
}


/*
double quadratic_interpolation (scalar uc, coord normalPoint, int type)
{
    double xc[3] = {0}; // x representing x or y depending on the type
    double v[3] = {0};
    double xp = type == 0? normalPoint.y: normalPoint.x;

    foreach_point (normalPoint.x, normalPoint.y) {
        for (int i = -1; i <= 1; i++) {
            if (type == 0)
                xc[i+1] = y + i*Delta;
            else // type = 1
                xc[i+1] = x + i*Delta;
            v[i+1] = type == 0? uc[0,i]: uc[i];
        }
    }

    double interpolate = v[0]*((xp - xc[1])*(xp - xc[2]))/((xc[0] - xc[1])*(xc[0]-xc[2]));
    interpolate += v[1]*((xp - xc[0])*(xp - xc[2]))/((xc[1] - xc[0])*(xc[1] - xc[2]));
    interpolate += v[2]*((xp - xc[0])*(xp - xc[1]))/((xc[2] - xc[0])*(xc[2] - xc[1]));

    return interpolate;
}
*/


/*
double quadratic_interpolation (scalar uc, coord normalPoint, int type) {
    double xp = (type == 0) ? normalPoint.y : normalPoint.x;
    double xc[3] = {0}, v[3] = {0};
    int i = 0; // Loop index

    // Initialize local variables and then assign after the loop
    double local_xc[3] = {0};
    double local_v[3] = {0};

    // Use a serial loop instead of foreach to avoid modifying non-local variables
    for (i = -1; i <= 1; i++) {
        if (type == 0) {
            local_xc[i+1] = normalPoint.y + i * Delta;
        } else {
            local_xc[i+1] = normalPoint.x + i * Delta;
        }
        local_v[i+1] = (type == 0) ? uc[0,i] : uc[i];
    }

    // After the loop, copy local values to xc and v arrays
    for (int j = 0; j < 3; j++) {
        xc[j] = local_xc[j];
        v[j] = local_v[j];
    }

    // Now perform the interpolation
    double interpolate = v[0] * ((xp - xc[1]) * (xp - xc[2])) / ((xc[0] - xc[1]) * (xc[0] - xc[2]));
    interpolate += v[1] * ((xp - xc[0]) * (xp - xc[2])) / ((xc[1] - xc[0]) * (xc[1] - xc[2]));
    interpolate += v[2] * ((xp - xc[0]) * (xp - xc[1])) / ((xc[2] - xc[0]) * (xc[2] - xc[1]));

    return interpolate;
}
*/


double ibm_dirichlet_gradientv2 (Point point, scalar uc, coord n, coord markerCoord, double boundaryCondition)
{
    coord cellCenter = {x, y}, distance, normalPoint;
    double d[2], v[2];
    int type;
    for (int num = 0; num <= 1; num++) {
        foreach_dimension()
            distance.x = ((1 + num) * Delta * n.x) - (markerCoord.x - cellCenter.x);

        if (fabs(n.x) >= fabs(n.y)) {
            type = 0;
            normalPoint.x = cellCenter.x + sign(n.x)*(Delta * (1 + num));
            normalPoint.y = markerCoord.y + distance.y;
        }
        else {  // fabs(n.x) < fabs(n.y)
            type = 1;
            normalPoint.x = markerCoord.x + distance.x;
            normalPoint.y = cellCenter.y + sign(n.y)*(Delta * (1 + num));
        }

        v[num] = scalar_bilinear_interpolation(point, uc, normalPoint);
        // v[num] = quadratic_interpolation (uc, normalPoint, type);
        d[num] = distance(markerCoord.x - normalPoint.x, markerCoord.y - normalPoint.y);
    }
    
    double gradient = ((boundaryCondition - v[0])*(d[1]/d[0]) - (boundaryCondition - v[1])*(d[0]/d[1]));
    gradient /= d[1] - d[0]; 

    return gradient;
}


coord ibm_gradientv2 (Point point, vector u, coord markerCoord, coord n)
{
    coord dudn, boundaryCondition;

    marker_point(point, vof, &markerCoord);
    bilinear_interpolation(point, u, markerCoord, &boundaryCondition);

    foreach_dimension()
        dudn.x = ibm_dirichlet_gradientv2 (point, u.x, n, markerCoord, boundaryCondition.x);
         
    return dudn;
}


coord extrapolate_gradient (Point point, scalar s, coord markerCoord, coord n, vector v)
{

    double weight[5][5] = {0};
    double weightSum = 0.;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (s[i,j] == 0) {

                coord cellCenter = {x + Delta*i,y + Delta*j}, d; // how to get x and y from cell centers?
                foreach_dimension()
                    d.x = markerCoord.x - cellCenter.x;

                double distanceMag = distance (d.x, d.y);
                double normalProjection = (n.x * d.x) + (n.y * d.y);

                weight[i][j] = sq(distanceMag) * fabs(normalProjection);

                weightSum += weight[i][j];
            }
            else
                weight[i][j] = 0.;
        }
    }

    coord dudn = {0};

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            foreach_dimension()
                dudn.x += (weight[i][j]/weightSum) * v.x[i,j];
        }
    }

    return dudn;
}

double extrapolate_scalar (Point point, scalar s, coord markerCoord, coord n, scalar p)
{
    double weight[5][5] = {0};
    double weightSum = 0.;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (s[i,j] == 0) {

                coord cellCenter = {x + Delta*i,y + Delta*j}, d;
                foreach_dimension()
                    d.x = markerCoord.x - cellCenter.x;

                double distanceMag = distance (d.x, d.y);
                double normalProjection = (n.x * d.x) + (n.y * d.y);

                weight[i][j] = sq(distanceMag) * fabs(normalProjection);

                weightSum += weight[i][j];
            }
            else
                weight[i][j] = 0.;
        }
    }

    double pressure = 0;

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            pressure += (weight[i][j]/weightSum) * p[i,j];
        }
    }

    return pressure;
}

double extrapolate_scalarv2 (Point point, scalar s, coord markerCoord, coord n, scalar p)
{
    double weight[5][5] = {0};
    double weightSum = 0.;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (s[i,j] < 0.5) {

                coord cellCenter = {x + Delta*i,y + Delta*j}, d;
                foreach_dimension()
                    d.x = markerCoord.x - cellCenter.x;

                double distanceMag = distance (d.x, d.y);
                double normalProjection = (n.x * d.x) + (n.y * d.y);

                weight[i][j] = sq(distanceMag) * fabs(normalProjection);

                weightSum += weight[i][j];
            }
            else
                weight[i][j] = 0.;
        }
    }

    double pressure = 0;

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            pressure += (weight[i][j]/weightSum) * p[i,j];
        }
    }

    return pressure;
}


/*
void ibm_dirichlet_gradient (Point point, vector u, scalar vof, coord n, coord pm, coord vb, coord * val)
{
    double d[2] = {nodata, nodata};
    coord v[2];
    double x1, y1;
    for (int l = 0; l <= 1; l++) {
        if (fabs(n.x) >= fabs(n.y)) {
            int i = sign(n.x);
            x1 = (Delta)*((l + 1) + pm.x);
            y1 = sign(n.y)*fabs(atan(n.y/n.x)) * x1;
            d[l] = distance(x1, y1);
            int j = fabs(y1 + (Delta*pm.y)) < Delta/2? 0: y1 + (Delta*pm.y) > Delta/2? 1: -1;
            if (vof[i*(l + 1), j] < 0.5 && vof[i*(l + 1), j+1] < 0.5 && vof[i*(l + 1), j-1] < 0.5)
                foreach_dimension()
                    v[l].x = quadratic(y1, (u.x[i*(l+1),j]), (u.x[i*(l+1),j+1]), (u.x[i*(l+1), j-1]));
            else
                foreach_dimension()
                    v[l].x = 0.;
            printy (point, vof, u, n, d, v, x1, y1, i, j, l, pm);
        }
        else if (fabs(n.x) < fabs(n.y)) {
            int j = sign(n.y);
            y1 = (Delta)*((l + 1) - pm.y);
            x1 = y1 / (sign(n.x)*fabs(atan(n.y/n.x)));
            d[l] = distance(x1, y1);
            int i = fabs(x1 + (Delta*pm.x)) < Delta/2? 0: x1 + (Delta*pm.x) > Delta/2? 1: -1;
            if (vof[i, j*(l + 1)] < 0.5 && vof[i+1,j*(l + 1)] < 0.5 && vof[i-1,j*(l + 1)] < 0.5)
                foreach_dimension()
                    v[l].x = quadratic (x1, (u.x[i,j*(l+1)]), (u.x[i+1,j*(l+1)]), (u.x[i-1,j*(l+1)]));
            else
                foreach_dimension()
                    v[l].x = 0.;
            printy (point, vof, u, n, d, v, x1, y1, i, j, l, pm);
        }
    }
    coord gradient;
    foreach_dimension()
        gradient.x = (d[1]/d[0]*(vb.x - v[0].x) - (d[0]/d[1])*(vb.x - v[1].x))/(d[1] - d[0]);
    *val = gradient;
    printx (point, vof, u, n, d, v, gradient);
}


coord ibm_gradient (Point point, vector u, coord pm, coord n, coord vb, scalar vof)
{
    coord dudn;
    ibm_dirichlet_gradient (point, u, vof, n, pm, vb, &dudn);
    if (dudn.x == nodata)
        dudn.x = -0.;
    return dudn;
}
*/
















/*
foreach_dimension()
static inline double dirichlet_gradient_x (Point point, scalar s, scalar vof, face vector sf,
                                           coord n, coord p, double bc, double * coef)
{
    foreach_dimension()
        n.x = - n.x;
    double d[2], v[2] = {nodata,nodata};
    bool defined = true;
    foreach_dimension()
        if (defined && !sf.x[(n.x > 0.)])
            defined = false;
    if (defined)
        for (int l = 0; l <= 1; l++) {
            int i = (l + 1)*sign(n.x);
            d[l] = (i - p.x)/n.x;
            double y1 = p.y + d[l]*n.y;
            int j = y1 > 0.5 ? 1 : y1 < -0.5 ? -1 : 0;
            y1 -= j;
#if dimension == 2
            if (sf.x[i + (i < 0),j] && sf.y[i,j] && sf.y[i,j+1] &&
                vof[i,j-1] && vof[i,j] && vof[i,j+1])
                v[l] = quadratic (y1, (s[i,j-1]), (s[i,j]), (s[i,j+1]));
#endif         
        }
        if (v[0] == nodata) {
            d[0] = max(1e-3, fabs(p.x/n.x));
            *coef = - 1./(d[0]*Delta);
            return bc/(d[0]*Delta);
        }
    *coef = 0.;
    if (v[1] != nodata) // third-order gradient
        return (d[1]*(bc - v[0])/d[0] - d[0]*(bc - v[1])/d[1])/((d[1] - d[0])*Delta);
    return (bc - v[0])/(d[0]*Delta); // second-order gradient
}

double dirichlet_gradient (Point point, scalar s, scalar vof, face vector sf,
			               coord n, coord p, double bc, double * coef)
{
#if dimension == 2
  foreach_dimension()
    if (fabs(n.x) >= fabs(n.y))
      return dirichlet_gradient_x (point, s, vof, sf, n, p, bc, coef);
#endif
  return nodata;
}
*/
