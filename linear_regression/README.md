## 1.	Time comparison (element-wise version vs. vectorized version)

Element wise(for-loops) :  0.17304396629333496 s 
Vectorized version : 0.0046100616455078125 s

Vectorized version is approximately 37.5x faster than element wise method.

## 2.	Estimated unknown function parameters W & b 
  
Set initial value as zero. (1e-6)

After 100 iterations,

w1 = 0.04378403, 
w2 = 0.04495862, 
b = -0.0013265678545876535

## 3.	Empirically determined (best) hyper parameter, 𝛼

𝛼 is known as the learning rate. We have to adjust it empirically. 

If 𝛼 is large, it could cause the overshooting. If it is small, it will require too many iterations to find a minimum gradient.

Basically, set 𝛼 = 0.01 initially and fine tune its value.
When 𝛼 = 0.0001, it seems that it increases the accuracy.
