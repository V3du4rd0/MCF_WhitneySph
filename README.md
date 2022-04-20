# Mean Curvature Flow  (MCF)

### Whitney Spheres evolving under MCF

Let _W_ be a Whitney Sphere inmersed in <img src="https://latex.codecogs.com/svg.image?\mathbb{C}^{m}"> and <img src="https://latex.codecogs.com/svg.image?z(s)">  its profile curve with normal vector <img src="https://latex.codecogs.com/svg.image?\mathbf{n}"> and curvature <img src="https://latex.codecogs.com/svg.image?\kappa&space;">. Let <img src="https://latex.codecogs.com/svg.image?p=\left<&space;z,\mathbf{n}\right>/|z|^2">, the evolution of <img src="https://latex.codecogs.com/svg.image?z(s)"> under the Mean Curvature Flow obeys the equation

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;z}{\partial&space;t}=(\kappa&space;&plus;&space;(m&plus;1)p)\mathbf{n}">

This equation exhibits some similarities with the standard MCF in <img src="https://latex.codecogs.com/svg.image?\mathbb{R}^{2}">. Therefore, we can adapt our MCF solver ([link](https://github.com/V3du4rd0/AMCF)) for planar curves to solve the previous equation. 

* Whitney Sphere's profile curve is a lemniscate. This one is a two loops curve whose parameter is defined on <img src="https://latex.codecogs.com/svg.image?\left&space;(&space;-\pi&space;,0&space;\right&space;)\cup&space;\left&space;(&space;0,\pi&space;&space;\right&space;)">. Then, we can compute each loop evolution independently.

* The `mod_amcf_14.py` file in our MCF solver ([link](https://github.com/V3du4rd0/AMCF)) containts all the _kernel_ functions to solve the MCF numerically. We modify the `TNK` function by adding the <img src="https://latex.codecogs.com/svg.image?\left&space;(&space;m&plus;1&space;\right&space;)p"> term to the curvature.

* Our previous [repo](https://github.com/V3du4rd0/AMCF) main file (`ImageRec_v4.py`) applies the MCF to a contour parametrization problem in high-contrast digital pictures. Here we present a script that avoids the high-contrast picture input and uses one of the lemniscate loops as initial curve.



<img src="https://user-images.githubusercontent.com/36924228/164336193-e8dfea72-3772-4009-8ae8-5575bcc9fcf3.gif" width="320" height="240" />
