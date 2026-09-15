# Plan 
here is the plan for what we are tyring to do from the context.md file you have seen the curves we are 
trying to create the following objects to work in this project. 

# Parent object 
each object related to the mathmatics of this porjct should have metadata attribuite 
e.g Curve.metadata should giveus that and we have the ablity to print it in a good fashion 
every attr of the object so you should think of this that the Parent object should include 
quality of life and DX things. the metadata would be a dict key value like but it can be nested 
for examole curve.metadata['certificat'] = "numerical" or curve.metadata['ploting'] = {'Color':'red','style':'dashed'}
the name can be changed as well 
this has to help us in the process to aribarly add new attrs to the already defind stuff
it should be optinal but we can retroactivly add to it. 
# Curve object 
now from the parent thing derive Curve they have defining implicit function like F(mu,c) = somthing 
evan some of them are the from mu = 2/5. with a name curve.name = "Delta_Q"

# Constraint 
this should capter something like 2/5 < mu < 1 something like that you should incroprate inf as well we can stick to numpy stuff 


# Point 
i dont knwo if this is necasry but you can have this mu = 1 and c = 0 eg

# Intersection 
curves = (C_1,C_2) intersection of two curves then the point in mu,c plane like Point(0,0) and metadata 

# Arc object 
this is a peice of a curve so we have curve=C_1 start_point end_point and maby a constrant optinal 

# Region 
This is the most imporatant thing here and we should build it well this a region defind by the sign of the some curves 
a name boundry arcs and vertices from interesections 

# Boudnry 
two regions R_1,R_2 the defind arc 
