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

import numpy as np
import math as m
    
def Construct1RDM_groundstate( solutionRHF, numPairs ):

    DoublyOccOrbs = solutionRHF[:, 0:numPairs]
    GS_OneRDM = 2 * np.dot( DoublyOccOrbs , DoublyOccOrbs.T )
    return GS_OneRDM

def Construct1RDM_addition( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    # vector(l) = \sum_{alpha ### VIRTUAL ###} C_{l,alpha} C^+_{alpha,i} / ( omega - epsilon_alpha + I*eta)
    Vector  = 1.0 / ( omega - eigsRHF[ numPairs: ] + 1j*eta )
    Vector  = np.multiply( Vector , solutionRHF[ orbital_i, numPairs: ] )
    Vector  = np.matrix( np.dot( solutionRHF[ :, numPairs: ] , Vector.T ) )
    if (Vector.shape[0] > 1):
        Vector = Vector.T # Now certainly row-like matrix (shape = 1 x len(vector))
    Overlap = np.dot( Vector, Vector.H ).real[0,0] # A Hermitian scalar is always real-valued
    Matrix  = np.dot( Vector.H, Vector ).real
    
    addition1RDM = groundstate1RDM + Matrix / Overlap
    return addition1RDM
    
def Construct1RDM_removal( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    # vector(l) = \sum_{alpha ### OCCUPIED ###} C_{l,alpha} C^+_{alpha,i} / ( omega - epsilon_alpha + I*eta)
    Vector  = 1.0 / ( omega - eigsRHF[ 0:numPairs ] + 1j*eta )
    Vector  = np.multiply( Vector , solutionRHF[ orbital_i, 0:numPairs ] )
    Vector  = np.matrix( np.dot( solutionRHF[ :, 0:numPairs ] , Vector.T ) )
    if (Vector.shape[0] > 1):
        Vector = Vector.T # Now certainly row-like matrix (shape = 1 x len(vector))
    Overlap = np.dot( Vector, Vector.H ).real[0,0] # A Hermitian scalar is always real-valued
    Matrix  = np.dot( Vector.H, Vector ).real

    removal1RDM = groundstate1RDM - Matrix / Overlap # Minus sign!
    return removal1RDM
    
def ConstructMeanFieldLDOS( orbital_i, omega, eta, eigsRHF, solutionRHF, numPairs ):

    # GF = 2 * \sum_{alpha all} C_{i,alpha} C^+_{alpha,i} / ( omega - epsilon_alpha + I*eta)
    GreenFunction = 2.0 * np.sum( np.multiply( 1.0 / ( omega - eigsRHF[ : ] + 1j*eta ) , np.square( solutionRHF[ orbital_i, : ] ) ) )
    LDOS = - GreenFunction.imag / m.pi
    return LDOS
    
def Construct1RDM_forward( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    # matrix(alpha,beta) = C^+_{alpha,i} C_{i,beta} / ( omega - ( epsilon_alpha - epsilon_beta ) + I*eta )
    nRows, nCols = solutionRHF.shape
    Matrix = np.zeros([ nRows - numPairs, numPairs ], dtype=complex) # First index (alpha) virtual, second index (beta) occupied
    for virt in range(numPairs, nRows):
        for occ in range(0, numPairs):
            Matrix[ virt - numPairs , occ ] = solutionRHF[ orbital_i, virt ] * solutionRHF[ orbital_i, occ ] / ( omega - eigsRHF[ virt ] + eigsRHF[ occ ] + 1j*eta )
    Matrix = np.matrix( Matrix )
    
    Overlap = 2 * np.trace( np.dot( Matrix , Matrix.H ) ).real
    Matrix1 = 2 * np.dot( solutionRHF[ : , numPairs:  ] , np.dot( np.dot( Matrix, Matrix.H ) , solutionRHF[ : , numPairs:  ].T ) ).real
    Matrix2 = 2 * np.dot( solutionRHF[ : , 0:numPairs ] , np.dot( np.dot( Matrix.H, Matrix ) , solutionRHF[ : , 0:numPairs ].T ) ).real
    
    forward1RDM = groundstate1RDM + ( Matrix1 - Matrix2 ) / Overlap
    return forward1RDM
    
def Construct1RDM_backward( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    return Construct1RDM_forward( orbital_i, -omega, -eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs )
    
def ConstructBathOrbitals( impurityOrbs, OneRDM, numBathOrbs ):

    embeddingOrbs = 1 - impurityOrbs
    embeddingOrbs = np.matrix( embeddingOrbs )
    if (embeddingOrbs.shape[0] > 1):
        embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
    isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
    numEmbedOrbs = np.sum( embeddingOrbs )
    embedding1RDM = np.reshape( OneRDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

    numImpOrbs = np.sum( impurityOrbs )
    numTotalOrbs = len( impurityOrbs )
        
    eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
    idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    pureEnvironEigVals = -eigenvals[numBathOrbs:]
    pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
    idx = pureEnvironEigVals.argsort()
    eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
    pureEnvironEigVals = -pureEnvironEigVals
    DiscardOccupation = np.sum( np.fabs( np.minimum( pureEnvironEigVals, 2.0 - pureEnvironEigVals ) ) )
    NelecEnvironment  = np.sum( pureEnvironEigVals ) # Number of electrons which are not on impurity or bath orbitals (float!)
    
    for counter in range(0, numImpOrbs):
        eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
    counter = 0
    for counter2 in range(0, numTotalOrbs):
        if ( impurityOrbs[counter2] ):
            eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
            eigenvecs[counter2, counter] = 1.0
            counter += 1
    assert( counter == numImpOrbs )
    
    # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
    assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

    # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
    # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
    # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
    return ( eigenvecs, NelecEnvironment, DiscardOccupation )

