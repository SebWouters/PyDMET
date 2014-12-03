'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

# Minimize the RMS difference of the correlated and mean-field 1-RDMs by
# changing the DMET potential on the impurity sites in the mean-field problem

import numpy as np
import LinalgWrappers
from scipy.optimize import minimize

def UmatFlat2Square( umatflat, lindim ):

    umatsquare = np.zeros( [ lindim, lindim ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatsquare[row, col] = umatflat[ row + ( col * ( col + 1 ) ) / 2 ]
            umatsquare[col, row] = umatsquare[row, col]
    return umatsquare
            
def UmatSquare2Flat( umatsquare, lindim ):

    umatflat = np.zeros( [ ( lindim * ( lindim + 1 ) ) / 2 ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatflat[ row + ( col * ( col + 1 ) ) / 2 ] = umatsquare[row, col]
    return umatflat

def OneRDMdifference( umatsquare, OneRDMcorr, HamDMET, numPairs ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    eigenvals, eigenvecs = LinalgWrappers.SortedEigSymmetric( HamDMET.Tmat )
    if ( eigenvals[ numPairs ] - eigenvals[ numPairs-1 ] < 1e-8 ):
        print "ERROR: The single particle gap is zero!"
        assert( eigenvals[ numPairs ] - eigenvals[ numPairs-1 ] >= 1e-8 )
    OneRDMmf = 2 * np.dot( eigenvecs[ :, 0:numPairs ] , eigenvecs[ :, 0:numPairs ].T )
    return OneRDMcorr - OneRDMmf

def CostFunction( umatflat, OneRDMcorr, HamDMET, numPairs ):

    umatsquare = UmatFlat2Square( umatflat, HamDMET.numImpOrbs )
    error = OneRDMdifference( umatsquare, OneRDMcorr, HamDMET, numPairs )
    return np.linalg.norm( error )**2

def Minimize( umat_guess, OneRDMcorr, HamDMET, NelecActiveSpace ):

    umatflat = UmatSquare2Flat( umat_guess, HamDMET.numImpOrbs )

    numPairs = NelecActiveSpace / 2
    result = minimize( CostFunction, umatflat, args=(OneRDMcorr, HamDMET, numPairs), options={'disp': False} )
    if ( result.success==False ):
        print "Minimize ::",result.message
    print "Minimize :: Cost function after convergence =",result.fun
    
    umatsquare = UmatFlat2Square( result.x, HamDMET.numImpOrbs )
            
    return umatsquare
    
    
