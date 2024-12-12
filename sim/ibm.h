#include "fractions.h"
#include "ibm-utils.h"


extern coord vc;          // object's imposed velocity
extern scalar vof;
extern int maxlevel;

vector desiredForce[];    // force calculated at the marker point
vector cellForce[];       // force located at the cell center (after spreading)
vector markerCoord[];     // field to store the coordinates of all marker points

vector velocityGrad[];

face vector faceForce[];  // for averaging the cell force to get the face values

vector utemp[];
vector forceTotal[];

int Ni = 10;               // # of multi-direct forcing iterations


event init (t = 0)
{
    foreach()
        foreach_dimension() { 
            velocityGrad.x[] = 0.;
                if (vof[] == 1)
                    u.x[] = vc.x;
        }
}

event acceleration (i++)
{
    trash({cellForce, desiredForce, markerCoord, faceForce, utemp});
    // 1. Get temporary velocity (advection, diffusion, pressure)
    foreach() {
        foreach_dimension() {
            utemp.x[] = u.x[] + dt * (g.x[] - forceTotal.x[]);
            forceTotal.x[] = 0.;
        }
    }

    for (int counter = 0; counter < Ni; counter++) { 

        // 2. calculate the force at the marker point
        foreach() {

            coord markerVelocity = {0}, desiredVelocity, markerPoint;

            if (vof[] > 0 && vof[] < 1) {

                marker_point (point, vof, &markerPoint);

                // interpolate to find velocity at marker point
                foreach_neighbor() {
#if dimension == 3                
                    double delta_u = delta_func (x, y, markerPoint.x, markerPoint.y, Delta, z, markerPoint.z);
#else
                    double delta_u = delta_func (x, y, markerPoint.x, markerPoint.y, Delta);
#endif                    
                    foreach_dimension()
                        markerVelocity.x += utemp.x[] * delta_u * dv();
                }

                // calculate the desired force at the marker point
                desiredVelocity = vc;
                foreach_dimension() {
                    desiredForce.x[] = (desiredVelocity.x - markerVelocity.x) / dt;
                    markerCoord.x[] = markerPoint.x;
                }
            }
            else if (empty_neighbor(point, &markerPoint, vof)) {
                foreach_neighbor() {
#if dimension == 3                
                    double delta_u = delta_func (x, y, markerPoint.x, markerPoint.y, Delta, z, markerPoint.z);
#else
                    double delta_u = delta_func (x, y, markerPoint.x, markerPoint.y, Delta);
#endif                    
                    foreach_dimension()
                        markerVelocity.x += utemp.x[] * delta_u * dv();
                }
                coord desiredVelocity = vc;
                foreach_dimension() {
                    desiredForce.x[] = (desiredVelocity.x - markerVelocity.x) / dt;
                    markerCoord.x[] = markerPoint.x;
                }
            }
            else
                foreach_dimension()
                    desiredForce.x[] = markerCoord.x[] = 0.;
        }

        // 3. spread the force at the marker point to the nearby cell centers
        foreach() {
            coord forceSum = {0};
            if (level == maxlevel) {
                double x1 = x, y1 = y;
#if dimension == 3
                double z1 = z;
#endif
                  foreach_neighbor()
                    if (markerCoord.x[] && level == maxlevel) {
#if dimension == 3                
                        double delta_h = delta_func (x1, y1, markerCoord.x[], markerCoord.y[], Delta, z1, markerCoord.z[]);
#else
                        double delta_h = delta_func (x1, y1, markerCoord.x[], markerCoord.y[], Delta);
#endif                    
                        foreach_dimension() {
                            forceSum.x += (desiredForce.x[] * delta_h * dv());
                        }
                    }
            }
            foreach_dimension() 
                cellForce.x[] = forceSum.x;
        }

        foreach()
            foreach_dimension() {
                forceTotal.x[] += cellForce.x[];
                utemp.x[] += dt*cellForce.x[];
            }
    }
    
    // 4. correct interfacial velocity
    foreach_face()
        faceForce.x[] = (face_value (forceTotal.x, 0));
    a = faceForce;


}

//  g is used to find uf t+dt/2 at the next time step, so the contributions
//  from f (stored in a) should be subtracted

event end_timestep (i++)
{
    trash({a});
    centered_gradient (p, g);

    trash ({velocityGrad});
    foreach()
        foreach_dimension()
            velocityGrad.x[] = (u.x[1] - u.x[-1])/(2.*Delta);
}

coord ibm_force ()
{
    coord ibmForce = {0};
    foreach(reduction(+:ibmForce))
        foreach_dimension()
            ibmForce.x += -forceTotal.x[]*dv();
    return ibmForce;
}

double ibm_pressure (Point point, scalar vof, scalar pressure, coord normal, coord markerPoint)
{
    return extrapolate_scalar (point, vof, markerPoint, normal, pressure);
}

/*
double ibm_vorticity (Point point, vector u, coord p, coord n) // needs improvement
{
    coord dudn = ibm_gradientv2 (point, u, p, n);

    return dudn.y*n.x - dudn.x*n.y;
}
*/
