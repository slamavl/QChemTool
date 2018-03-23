#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "nabcode.h"
extern char NAB_rsbuf[];
static int mytaskid, numtasks;

static MOLECULE_T *m;

static REAL_T x[990], fret;


int main( argc, argv )
	int	argc;
	char	*argv[];
{
	nabout = stdout; /*default*/

	mytaskid=0; numtasks=1;
static INT_T __it0001__;
static INT_T __it0002__;
static INT_T __it0003__;
static INT_T __it0004__;
static REAL_T __ft0001__;
static REAL_T __ft0002__;
m = getpdb( "nab_input.pdb", NULL );
readparm( m, "nab_input.prmtop" );
mm_options( "cut=999.0, ntpr=50" );
setxyz_from_mol(  &m, NULL, x );
mme_init( m, NULL, "::ZZZ", x, NULL );


conjgrad( x, ITEMP( __it0001__, 3 *  *( NAB_mri( m, "natoms" ) ) ),  &fret, mme, FTEMP( __ft0001__, 1.000000E-03 ), FTEMP( __ft0002__, 1.000000E-04 ), ITEMP( __it0002__, 30000 ) );


mm_options( "ntpr=10" );
newton( x, ITEMP( __it0001__, 3 *  *( NAB_mri( m, "natoms" ) ) ),  &fret, mme, mme2, FTEMP( __ft0001__, 1.000000E-08 ), FTEMP( __ft0002__, 0.000000E+00 ), ITEMP( __it0002__, 500 ) );


setmol_from_xyz(  &m, NULL, x );
putpdb( "nab_opt_structure.pdb", m, NULL );


nmode( x, ITEMP( __it0001__, 3 *  *( NAB_mri( m, "natoms" ) ) ), mme2, ITEMP( __it0002__, 990 ), ITEMP( __it0003__, 0 ), FTEMP( __ft0001__, 0.000000E+00 ), FTEMP( __ft0002__, 0.000000E+00 ), ITEMP( __it0004__, 0 ) );








	exit( 0 );
}
