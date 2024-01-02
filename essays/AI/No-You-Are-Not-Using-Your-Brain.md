# 💎 No, you are not using your brain!

Let's face it: passively watching tutorials or skimming books won't help you in the long run. Why? Because you tend not to use your brain at all during the process. If you don't feel your brain heating up to the point of pain, for example, while trying to grasp linear algebra, you're not using any brain power. Thus, your time is wasted.

In the context of linear algebra, when looking at a system of equations, the coefficients form a matrix A, the unknowns are in a vector x, and the constant terms are in another vector b. The equations themselves are dot product notations. This can be represented as Ax = b, and the objective is to solve for x.

If A is invertible or non-singular, it means that the matrix A has an inverse A^-1. This implies that there exists a unique solution for the linear system Ax = b. The solution can be expressed as x = A^-1 * b. The inverse matrix A^-1 has the property that when it's multiplied with A, it produces the identity matrix, AA^-1 = I. Therefore, by multiplying A^-1 with b, we can find x, which will be the unique solution to the system.

Eigenvalues and eigenvectors are crucial elements that represent the characteristics of matrix A. Simply put, when you multiply a matrix A by its eigenvector, the direction of the vector doesn't change; it only gets scaled by the eigenvalue. Mathematically, this is expressed as Ax=λx, where λ is the eigenvalue and x is the eigenvector.

The most difficult concepts to grasp are the eigen-s*it, the final boss of the journey. In the simplest terms that even a child could understand: the eigenvector is like a "rubber band with a fixed direction," and the eigenvalue is like the "ruler that stretches or shrinks that rubber band by its exact length." You don't have to dive any deeper into the eigen-hell. Trust me.

These concepts are fundamental not only in linear algebra but also in various fields such as optimization, machine learning, and signal processing. Essentially, these concepts are all we need to grasp in linear algebra when it comes to learning AI.

If you understand the basics and apply them using numpy, sympy, etc, that's enough. You'll never solve these complex problems using pen and paper. Don't overdo it.

https://www.coursera.org/learn/machine-learning-linear-algebra/

📚 Recommended reference book:

Practical Linear Algebra for Data Science: From Core Concepts to Applications Using Python - https://www.amazon.com/Practical-Linear-Algebra-Data-Science/dp/1098120612
