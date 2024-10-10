ts = 1/8;
nmax = 8;
n= 0:1:nmax -1;

A = 1;
T0 = 1;
F0 = 1/T0;
theta = 0;
D = 0;

xn = A *sin(2 * pi * F0 *  n/nmax+ theta ) + D
tsa = 1/1000;
t = 0:tsa:1;
xt = A *sin(2 * pi * F0 * t + theta ) + D;
