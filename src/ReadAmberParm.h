#ifndef READ_AMBER_PARM_H
#define READ_AMBER_PARM_H

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <functional>

/* Full specification:
 * https://ambermd.org/FileFormats.php#topology 
 * https://ambermd.org/prmtop.pdf
 * 
 * Requirements for prmtop parsers Parsers: regardless of the language they are 
 * written in, should conform to a list of attributes to maximize the likelihood 
 * that they are parsed correctly. 
 * 1. Parsers should expect that some 4-character fields (e.g., atom or residue names) 
 *    may have some names that have 4 characters and therefore might not be 
 *    whitespace-delimited. 
 * 2. Parsers should not expect SECTIONs in the prmtop to be in any particular 
 *    order. 
 * 3. Parsers should not expect or require %COMMENT lines to exist, but should 
 *    properly parse the file if any number of %COMMENT lines appear as indicated 
 *    above 
 * 4. The topology file may be assumed to have been generated ‘correctly’ by tleap or 
 *    some other credible source. No graceful error checking is required.
 * 
 * To meet the requirement 1, this implementation uses a two-stage approach. The first 
 * stage splits the files by sections containing %FLAG, %FORMAT and data into a map, 
 * and it doesn't care about the actual meaning of each section.
 * The second stage analyses the %POINTER section at first, and then turns the map into
 * a struct used by NAMD. This stage will match the section names by hard-coded strings, 
 * and do lots of sanity checks and memory allocations.
 */

namespace AmberParm7Reader {

#ifdef DOUBLE
#define _REAL		double
#else
#define _REAL		float
#endif

using std::vector;
using std::string;
using std::unordered_map;
using std::tuple;
using std::make_tuple;

/* Frankly speaking, I don't fully understand the whole specification of fortran 
 * format specifiers. I try to implement a few of them, which should be enough for 
 * parsing the AMBER parm7 format.
 */
struct FortranFormatSpecifier {
  bool IsPadding; // boolean value to determine if this is a specifier or a string padding
  string Padding; // support padding characters inside quotation marks
  char Type;      // type of the field
  int Width;      // the length of the field
  int NumOfFields;// number of fields
  int Precision;  // precision of real type
};

/* struct for reading fortran data regardless of types
 * I'd like to use std::variant but that is in C++17 and we are bound to C++11
 */
struct FortranData {
  enum class FortranDataType {String, Int, Real};
  FortranDataType Type;
  string String;
  int Int;
  _REAL Real;
  FortranData() {}
  // constructors by types
  FortranData(const string& str): Type(FortranDataType::String), String(str) {}
  FortranData(const int& x): Type(FortranDataType::Int), Int(x) {}
  FortranData(const _REAL& x): Type(FortranDataType::Real), Real(x) {}
};

/* A map to store the data read from AMBER parm7 format
 * key: title of a %FLAG section
 * value: a tuple of a boolean and the read data from the section
 *        the boolean value is used in read_amber_parm_stage2 to check if the 
 *        section is really parsed
 */
typedef unordered_map<string, tuple<bool, vector<FortranData>>> AmberTopparMap;

/* AMBER data structure used by NAMD
 */
struct Ambertoppar {
  bool HasData = false;
  bool IsCharmmFF; // is this a CHARMM force field in AMBER format?
  // version info
  string build_time;
  string version_stamp;
  string ititl; // title
  int Natom;    // total number of atoms
  int Ntypes;   // total number of distinct atom types
  int Nbonh;    // number of bonds containing hydrogen
  int Mbona;    // number of bonds not containing hydrogen
  int Ntheth;   // number of angles containing hydrogen
  int Mtheta;   // number of angles not containing hydrogen
  int Nphih;    // number of dihedrals containing hydrogen
  int Mphia;    // number of dihedrals not containing hydrogen
  int Nhparm;   // currently not used
  int Nparm;    // used to determine if addles created prmtop
  int Nnb;      // number of excluded atoms
  int Nres;     // number of residues
  int Nbona;    // MBONA + number of constraint bonds
  int Ntheta;   // MTHETA + number of constraint angles
  int Nphia;    // MPHIA + number of constraint dihedrals
  int Numbnd;   // number of unique bond types
  int Numang;   // number of unique angle types
  int Nptra;    // number of unique dihedral types
  int Natyp;    // number of atom types in parameter file, see SOLTY below
  int Nphb;     // number of distinct 10-12 hydrogen bond pair types
  int Ifpert;   // set to 1 if perturbation info is to be read in
  int Nbper;    // number of bonds to be perturbed
  int Ngper;    // number of angles to be perturbed
  int Ndper;    // number of dihedrals to be perturbed
  int Mbper;    // number of bonds with atoms completely in perturbed group
  int Mgper;    // number of angles with atoms completely in perturbed group
  int Mdper;    // number of dihedrals with atoms completely in perturbed groups
  int Ifbox;    // set to 1 if standard periodic box, 2 when truncated octahedral
  int Nmxrs;    // number of atoms in the largest residue
  int Ifcap;    // set to 1 if the CAP option from edit was specified
  int Numextra; // number of extra points found in topology
  int Ncopy;    // number of PIMD slices / number of beads
  // parameter data (copied from parm.C)
  vector<string> AtomNames;       // (size: Natom) names
  vector<_REAL>  Charges;         // (size: Natom) charges
  vector<_REAL>  Masses;          // (size: Natom) masses
  vector<int>    AtomNumbers;     // (size: Natom) atomic numbers
  vector<int>    Iac;             // (size: Natom) LJ atom type indices
  vector<int>    Iblo;            // (size: Ntypes * Ntypes) the number of atoms excluded from NB calculation
  vector<int>    Cno;             // (size: Ntypes * Ntypes) LJ atom types
  vector<string> ResNames;        // (size: Nres) residue names
  vector<int>    Ipres;           // (size: Nres) the first atoms in residues
  vector<_REAL>  Rk;              // (size: Numbnd) bond force constants
  vector<_REAL>  Req;             // (size: Numbnd) bond equilibrium distances
  vector<_REAL>  Tk;              // (size: Numang) angle force constants
  vector<_REAL>  Teq;             // (size: Numang) angle equilibrium angles
  vector<_REAL>  Pk;              // (size: Nptra) dihedral angle force constants
  vector<_REAL>  Pn;              // (size: Nptra) dihedral angle periodicities
  vector<_REAL>  Phase;           // (size: Nptra) dihedral angle phases
  vector<_REAL>  SceeScaleFactor; // (size: Nptra) factors by which 1-4 electrostatic interactions are divided
  vector<_REAL>  ScnbScaleFactor; // (size: Nptra) factors by which 1-4 VdW interactions are divided
  vector<_REAL>  Solty;           // (size: Natyp) unused
  vector<_REAL>  Cn1;             // (size: Ntypes * (Ntypes + 1) / 2) coefficient As of LJ interactions
  vector<_REAL>  Cn2;             // (size: Ntypes * (Ntypes + 1) / 2) coefficient Bs of LJ interactions
  vector<int>    BondHAt1;        // (size: Nbonh) atom(1) indices in bonds involving hydrogens
  vector<int>    BondHAt2;        // (size: Nbonh) atom(2) indices in bonds involving hydrogens
  vector<int>    BondHNum;        // (size: Nbonh) bond indices involving hydrogens
  vector<int>    BondAt1;         // (size: Nbona) atom(1) indices in bonds not involving hydrogens
  vector<int>    BondAt2;         // (size: Nbona) atom(2) indices in bonds not involving hydrogens
  vector<int>    BondNum;         // (size: Nbona) bond indices not involving hydrogens
  vector<int>    AngleHAt1;       // (size: Ntheth) atom(1) indices in angles involving at least one H
  vector<int>    AngleHAt2;       // (size: Ntheth) atom(2) indices in angles involving at least one H
  vector<int>    AngleHAt3;       // (size: Ntheth) atom(3) indices in angles involving at least one H
  vector<int>    AngleHNum;       // (size: Ntheth) angle indices involving at least one H
  vector<int>    AngleAt1;        // (size: Ntheta) atom(1) indices in angles not involving H
  vector<int>    AngleAt2;        // (size: Ntheta) atom(2) indices in angles not involving H
  vector<int>    AngleAt3;        // (size: Ntheta) atom(3) indices in angles not involving H
  vector<int>    AngleNum;        // (size: Ntheth) angle indices not involving H
  vector<int>    DihHAt1;         // (size: Nphih) atom(1) indices in dihedral angles involving at least one H
  vector<int>    DihHAt2;         // (size: Nphih) atom(2) indices in dihedral angles involving at least one H
  vector<int>    DihHAt3;         // (size: Nphih) atom(3) indices in dihedral angles involving at least one H
  vector<int>    DihHAt4;         // (size: Nphih) atom(4) indices in dihedral angles involving at least one H
  vector<int>    DihHNum;         // (size: Nphih) dihedral angles indices involving at least one H
  vector<int>    DihAt1;          // (size: Nphih) atom(1) indices in dihedral angles not involving H
  vector<int>    DihAt2;          // (size: Nphih) atom(2) indices in dihedral angles not involving H
  vector<int>    DihAt3;          // (size: Nphih) atom(3) indices in dihedral angles not involving H
  vector<int>    DihAt4;          // (size: Nphih) atom(4) indices in dihedral angles not involving H
  vector<int>    DihNum;          // (size: Nphih) dihedral angles indices not involving H
  vector<int>    ExclAt;          // (size: Nnb) a list for each atom of excluded partners in the NB calculation
  vector<_REAL>  HB12;            // (size: Nphb) coefficient As in the 12-10 potential
  vector<_REAL>  HB10;            // (size: Nphb) coefficient Bs in the 12-10 potential
  vector<string> AtomSym;         // (size: Natom) atom type names for all atoms
  vector<string> AtomTree;        // (size: Natom) the tree structure of each atom
  vector<int>    TreeJoin;        // (size: Natom) unused
  vector<int>    AtomRes;         // (size: Natom) unused
  // solvent pointers (if Ifbox > 0)
  int Iptres; // the final residue that is part of the solute
  int Nspm;   // the total number of molecules
  int Nspsol; // the first solvent molecules
  vector<int>    AtomsPerMolecule;// (size: Nspm) how many atoms are present in each molecule
  vector<_REAL>  BoxDimensions;   // (size: 4) box angle and dimensions
  vector<int>    CapInfo;         // (size: 1) the last atom before the water cap begins
  vector<_REAL>  CapInfo2;        // (size: 4) distacne from cap center to outside the cap, coordinates of the cap center
  vector<string> RadiusSet;       // (size: 1) intrinsic implicit solvent radii set
  vector<_REAL>  Radii;           // (size: Natom) intrinsic radii of atoms
  vector<_REAL>  Screen;          // (size: Natom) screening parameters used in Generalized Born
  // polarizability
  int Ipol;     // 0: fixed-charge force fields, 1: force fields that contain polarization
  vector<_REAL>  Polarizability;  // (size: Natom) atomic polarizabilities
  // CHAMBER support (CHARMM FF in AMBER format)
  // Urey-Bradley terms
  int Nub;      // total number of Urey-Bradley terms
  int Nubtypes; // number of unique Urey-Bradley types
  vector<int>   UreyBradleyAt1;   // (size: Nub) atom(1) indices in UB terms
  vector<int>   UreyBradleyAt2;   // (size: Nub) atom(2) indices in UB terms
  vector<int>   UreyBradleyNum;   // (size: Nub) types of UB terms
  vector<_REAL> UreyBradleyForceConstants; // (size: Nubtypes) force constants of UB terms
  vector<_REAL> UreyBradleyEquilValues;    // (size: Nubtypes) equilibrium distances of UB terms
  // impropers
  int Nimphi;   // total number of improper terms
  vector<int>   ImproperAt1;      // (size: Nimphi) atom(1) indices in improper terms
  vector<int>   ImproperAt2;      // (size: Nimphi) atom(2) indices in improper terms
  vector<int>   ImproperAt3;      // (size: Nimphi) atom(3) indices in improper terms
  vector<int>   ImproperAt4;      // (size: Nimphi) atom(4) indices in improper terms
  vector<int>   ImproperNum;      // (size: Nimphi) types of improper terms
  int NimprTypes; // total number of the types of improper terms
  vector<_REAL> ImproperForceConstants; // (size: NimprTypes) force constants of improper terms
  vector<_REAL> ImproperPhases;         // (size: NimprTypes) phases of improper terms
  // CHARMM scaled 1-4 coefficients
  vector<_REAL> LJ14ACoefficients; // (size: Ntypes * (Ntypes + 1) / 2) LJ parameter As
  vector<_REAL> LJ14BCoefficients; // (size: Ntypes * (Ntypes + 1) / 2) LJ parameter Bs
  // CMAP data in AMBER ff19SB or CHARMM
  // CMAP meta data cannot be obtained from %FLAG POINTERS
  bool HasCMAP;       // CMAP is not always enabled, we need this flag to check it
  int  CMAPTermCount; // number of CMAP terms
  int  CMAPTypeCount; // number of CMAP types
  vector<int>           CMAPResolution; // (size: CMAPTypeCount) CMAP resolution of each type
  vector<vector<_REAL>> CMAPParameter;  // (size: CMAPTypeCount, CMAPResolution[i] * CMAPResolution[i]) CMAP grid parameters
  vector<int>           CMAPIndex;      // (size: CMAPTermCount * 6) atom indices and type of a CMAP term
};

/**
 *  @brief  Parse a single fortran format specifier.
 *  @details
 *   A good documentation for fortran format specifiers can be found at:
 *   https://www.obliquity.com/computer/fortran/format.html. However, I don't plan 
 *   to support all of them since that is too complicated and the AMBER parm7 files 
 *   only use a subset of them. Here's a list of examples of the specifiers:
 *     5E16.8: match 5 floating point numbers, with 16 as field width and 8 as precision.
 *     i2: match an integer with 2 as field width.
 *     a78: match a string with 78 characters.
 *     20a4: match 20 strings and each string has 4 characters.
 *   This function can parse the a string of above into a FortranFormatSpecifier.
 *  @param  str a specifier string.
 *  @param  specifier a fortran specifier struct.
 */
void parse_fortran_format(const std::string& str, FortranFormatSpecifier& specifier);

/**
 *  @brief Parse multiple fortran format specifiers separated by commas.
 *  @param  str a specifier string. For example: "i2,a78".
 *  @param  specifiers a vector of fortran specifier structs.
 */
void parse_fortran_format(const std::string& str, vector<FortranFormatSpecifier>& specifiers);

/**
 *  @brief This function splits a string by a set of fortran format specifiers.
 *  @details 
 *   This function splits a string by a set of fortran format specifiers. For example:
 *   If the source string is:
 *   "  920979       9  920959      19      47      25      85      66       0       0", 
 *   and there is only one specifier, "10I8" in the specifiers vector, then the vector 
 *   destination will be APPENDED with ["  920979", "       9", "  920959",...]
 *  @param source the source string.
 *  @param destination the destination container of the splited data.
 */
void split_string_by_specifiers(const string& source,
                                const vector<FortranFormatSpecifier>& specifiers,
                                vector<FortranData>& destination);

/**
 *  @brief  Read an AMBER topology file into an AmberTopparMap.
 *  @param  filename the filename of the AMBER topology file.
 *  @param  toppar_map a map contains the FLAG title and the data.
 *  @return read status. It will return 0 if the reading is successful.
 */
bool read_amber_parm_stage1(const char* filename, AmberTopparMap& toppar_map);

/**
 *  @brief  Read an AmberTopparMap into an Ambertoppar struct for NAMD.
 *  @param  toppar_map a map contains the FLAG title and the data
 *  @param  toppar_data an Ambertoppar struct for NAMD.
 *  @return read status. It will return 0 if the reading is successful.
 */
bool read_amber_parm_stage2(AmberTopparMap& toppar_map, Ambertoppar& toppar_data);

// helper function for read the POINTER section and allocate the memory 
// according to the sizes in it.
bool parse_pointer(const vector<FortranData>& source,
                   Ambertoppar& result);

/**
 *  @brief Copy data from source to destination.
 *  @param source source data from AmberTopparMap.
 *  @param count the number of fields. It should equal to the size of source.
 *  @param destination destination array in Ambertoppar.
 *  @param section_name name of the section (used for sanity check only).
 *  @return read status. It will return 0 if the reading is successful.
 */
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   vector<string>& destination,
                   const string& section_name);
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   vector<int>& destination,
                   const string& section_name);
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   vector<_REAL>& destination,
                   const string& section_name);

/**
 *  @brief Copy data from source to multiple destination arrays.
 *  @details
 *    Assuming the size of destination list is M, the size of the source array 
 *    should be M * count. This function copies source[M*i+j] to destination[j][i].
 *  @param source source data from AmberTopparMap.
 *  @param count the number of fields. It should equal to the size of source divided by the size of destination.
 *  @param destination multiple destination arrays in Ambertoppar.
 *  @param section_name name of the section (used for sanity check only).
 */
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   std::initializer_list<std::reference_wrapper<vector<string>>> destination,
                   const string& section_name);
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   std::initializer_list<std::reference_wrapper<vector<int>>> destination,
                   const string& section_name);
bool parse_section(const vector<FortranData>& source,
                   const int& count,
                   std::initializer_list<std::reference_wrapper<vector<_REAL>>> destination,
                   const string& section_name);

// reader function for NAMD
Ambertoppar readparm(const char* filename);

}

#endif
