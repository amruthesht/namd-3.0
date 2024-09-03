# NAMD

NAMD is a parallel, object-oriented molecular dynamics code designed for
high-performance simulation of large biomolecular systems.  It runs on laptops 
and desktops up through some of the world's largest supercomputers. 
NAMD is distributed free of charge and allows some amount of reuse of code 
as specified by the [NAMD license](license.txt) agreement. 

## Repository

The repository contains three protected branches:
- `main` - the default branch containing NAMD 3.x code
- `master` - the former default branch containing NAMD 2.x code
- `devel` - the former development branch that has been renamed `main`

The `master` and `devel` branches are now frozen in place. All future development 
will be towards extending `main`.

If you cloned the repository before `main` was created, you can run the following 
to rename your `devel` branch and set the correct upstream:
- `git fetch origin --prune && git branch -m devel main && git branch --set-upstream-to=origin/main main`

## Building

We strive to support NAMD on commodity hardware platforms. 
Please see [notes.txt](notes.txt) for details on how to build NAMD 
on various platforms. 

## Contributions

NAMD is developed as a research and teaching tool, and we encourage contributions 
to the code from the community.  Anyone who wishes to contribute to NAMD development 
should email [namd\@ks.uiuc.edu](mailto:namd@ks.uiuc.edu) to have repository access 
upgraded from `Guest` to `Developer`.  Please be aware of the [NAMD license](license.txt) 
as it pertains to code contributions. 

## Acknowledgments

Full documentation for using NAMD is available from the 
[NAMD website](https://www.ks.uiuc.edu/Research/namd/). 

The NAMD project is funded by the National Institutes of Health 
National Institute of General Medical Sciences 
(NIH P41-GM104601 and NIH R24-GM145965). 

Please also see the original [READE.txt](README.txt) file. 
