
TCLDIR=/Projects/namd2/tcl/tcl8.5.9-darwin-arm8-clang-threaded
TCLINCL=-I$(TCLDIR)/include
TCLLIB=-L$(TCLDIR)/lib -ltcl8.5
TCLFLAGS=-DNAMD_TCL
TCL=$(TCLINCL) $(TCLFLAGS)

