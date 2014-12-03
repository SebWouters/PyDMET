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

# This file contains two functions to compute the bath orbitals from the embedding RHF solution

import numpy as np

def ConstructEmbedding1RDM( impurityOrbs, solutionRHF, numPairs ):
    
    DoublyOccOrbs = solutionRHF[:, 0:numPairs]
    OneRDM = 2 * np.dot( DoublyOccOrbs , DoublyOccOrbs.T )
    environOrbs = 1 - impurityOrbs
    booleanMat = np.dot( np.matrix( environOrbs ).T , np.matrix( environOrbs ) ) == 1
    numEnvironOrbs = np.sum( environOrbs )
    embedding1RDM = np.reshape( OneRDM[ booleanMat ], ( numEnvironOrbs , numEnvironOrbs ) )
    return embedding1RDM

def ConstructBathOrbitals( impurityOrbs, embedding1RDM, numBathOrbs ):

    numImpOrbs = np.sum( impurityOrbs )
    if ( numBathOrbs == None ):
        numBathOrbs = numImpOrbs
    numTotalOrbs = len( impurityOrbs )
        
    eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
    idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    pureEnvironEigVals = -eigenvals[numBathOrbs:]
    pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
    idx = pureEnvironEigVals.argsort()
    eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
    
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
    return eigenvecs

