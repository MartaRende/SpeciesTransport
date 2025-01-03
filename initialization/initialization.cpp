#include <math.h>

#include "init.h"
#include <stdio.h>

using namespace std;

// Compute the triangle sign
double sign (const double x1, const double y1, const double x2, const double y2, const double x3, const double y3)
{
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
}

// Check if a point is inside a triangle
bool isInsideTriangle (const double x1, const double y1, const double x2, const double y2, const double x3, const double y3, const double x, const double y)
{
    double d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(x, y, x1, y1, x2, y2);
    d2 = sign(x, y, x2, y2, x3, y3);
    d3 = sign(x, y, x3, y3, x1, y1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

// Check if a point is inside a circle
bool isInsideCircle(const double x1, const double y1, const double radius, const double x, const double y ){
    double d = pow( (x1-x)*(x1-x) + (y1-y)*(y1-y) ,0.5);
    return d <= radius;
}

// Initialization of the temperature inside the domain
void Initialization(double** Y,double** u, double** v, const int nx, const int ny, const double dx, const double dy, const int s){
   // == Initailisation is the same of the lab of energy diffusion but each part of the logo ISC is a different Y
    // ISC LOGO
    // == Logo parameters ==
    double xcenter = 0.6; // Logo position x
    double ycenter = 0.65; // Logo position y
    double radius = 0.05; // Logo scale
    double size = sqrt(2)/2.0 * radius / 0.5;

    // == Colors -> temperature ==
    for (int i = 0; i < nx; i++){
        double x = i*dx - xcenter;    
        for (int j = 0; j < ny; j++){
            double y = j*dy - ycenter;

           if (isInsideCircle(0.0, 0.0, radius, x, y) && s == 0){ // White
                Y[i][j] = 1.0;
            } 
            else if ( (isInsideTriangle(0.0, -radius, 0.0, radius, size + radius, radius, x, y) or 
                      isInsideTriangle(0.0, -radius, size + radius, -radius, size + radius, radius, x, y) or 
                      isInsideCircle(size + radius, 0.0, radius, x, y)) && s == 1 ){ // Green
                Y[i][j] = 15.0;
            }
            else if ( (isInsideTriangle(size/2.0, -size/2.0, -size/2.0, size/2.0, radius, size + radius, x, y) or 
                      isInsideTriangle(size/2.0, -size/2.0, size + radius, radius, radius, size + radius, x, y) or 
                      isInsideCircle(size/2.0 + radius, size/2.0 + radius, radius, x, y)) && s ==2 ){ // Pink
                Y[i][j] = -5.0;
            }
            else if ( (isInsideTriangle(radius, 0.0, -radius, 0.0, -radius, size + radius, x, y) or 
                      isInsideTriangle(radius, 0.0, radius, size + radius, -radius, size + radius, x, y) or 
                      isInsideCircle(0.0, size + radius, radius, x, y)) && s == 3 ){ // Purple
                Y[i][j] = 10.0;
            }
            else if ( (isInsideTriangle(-size/2.0, -size/2.0, size/2.0, size/2.0, -radius, size + radius, x, y) or 
                      isInsideTriangle(-size/2.0, -size/2.0, -size - radius, radius, -radius, size + radius, x, y) or 
                      isInsideCircle(-size/2.0 - radius, size/2.0 + radius, radius, x, y)) && s ==  4){ // Blue
                Y[i][j] = 5.0;
            }
            else if ( (isInsideTriangle(0.0, -radius, 0.0, radius, -size - radius, radius, x, y) or 
                      isInsideTriangle(0.0, -radius, -size - radius, -radius, -size - radius, radius, x, y) or 
                      isInsideCircle(-size - radius,0.0,radius, x, y)) && s == 5){ // Yellow
                Y[i][j] = -10.0;
          
            }
            else { 
                Y[i][j] = 0.0;
            }
            // init of speeds only for the first species beacuse velocity are always the same
            if(s==0){
            u[i][j] = -sin(2.0*M_PI*j*dy) * sin(M_PI*i*dx) * sin(M_PI*i*dx);
            v[i][j] = sin(2.0*M_PI*i*dx) * sin(M_PI*j*dy) * sin(M_PI*j*dy);
            }
         

        }

    }


}
