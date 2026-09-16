# Curves 

## C1 
name: C1 
exper:
{4 - 12 c + 9 c^2 - 44 \[Mu] + 50 c \[Mu] - 24 c^2 \[Mu] + 
   121 \[Mu]^2 - 56 c \[Mu]^2 + 16 c^2 \[Mu]^2 == 0}

role: 'Delta of Q' 
you can add other stuff for color and what not 

## C2 
name: C2 
exper: 
-2 c + c^2 - 2 \[Mu] + 10 c \[Mu] + 5 \[Mu]^2 - 16 c \[Mu]^2 == 0
role: 'collion of branch A,B'

## C3 
name: C3 
exper {\[Mu] == 2/5}
role: colion of branch A 

## C4 
exper: {c == 2 + \[Mu]}
role: collsion of B with 0,1 

## C5 
{c == 0}
role: 'denmanator singularity and collion branch B'

## C6 
{\[Mu] == 1/4}
role: 'x^2 degenarcy'

# Intersection 

## I1 
certificate: numerical 
C1 C2 
{{\[Mu] -> 0.191185, c -> 0.895806}, {\[Mu] -> 0.967111, c -> 6.89609}}

## I2 
cert: symbolic 
C1 C5 
{{\[Mu] -> 2/11, c -> 0}}

## I3 
cert:symbolic 
C1 C6 
{{\[Mu] -> 1/4, c -> 3/8}}

## I4 
cert: symbolic 
C2 C3 
{{\[Mu] -> 2/5, c -> 0}, {\[Mu] -> 2/5, c -> 14/25}}

## I5 
cert: symbolic 
C2 C4 
{{\[Mu] -> 0, c -> 2}, {\[Mu] -> -2 + 1/2 (3 - Sqrt[6]), 
  c -> 1/2 (3 - Sqrt[6])}, {\[Mu] -> -2 + 1/2 (3 + Sqrt[6]), 
  c -> 1/2 (3 + Sqrt[6])}}

## I6 
C2 C5 
cert: symbolic
{{\[Mu] -> 0, c -> 0}, {\[Mu] -> 2/5, c -> 0}}

## I7 
C2 C6 
cert: symbolic
{{\[Mu] -> 1/4, c -> -(1/4)}, {\[Mu] -> 1/4, c -> 3/4}}

## I8 
C3 C4 
cert: symbolic
{{\[Mu] -> 2/5, c -> 12/5}}

## I9 
C3 C5 
cert: symbolic
{{\[Mu] -> 2/5, c -> 0}}

## I10 
C4 C5 
cert: symbolic 
{{\[Mu] -> -2, c -> 0}}

## I11 
C4 C6 
cert: symbolic
{{\[Mu] -> 1/4, c -> 9/4}}


## I12 
C5 C6 
cert: symbolic 
{{\[Mu] -> 1/4, c -> 0}}
