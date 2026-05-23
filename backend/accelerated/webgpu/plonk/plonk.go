//go:build js && wasm

package plonk

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	csbn254 "github.com/consensys/gnark/constraint/bn254"
)

// Prove runs the PLONK prover for supported curves.
//
// For now only BN254 is supported. The implementation is a local copy of
// gnark's generated BN254 PLONK prover so we can replace individual prover
// phases with WebGPU calls without changing the browser-facing runtime API.
func Prove(spr constraint.ConstraintSystem, pk gnarkplonk.ProvingKey, fullWitness witness.Witness) (gnarkplonk.Proof, error) {
	switch typedSPR := spr.(type) {
	case *csbn254.SparseR1CS:
		typedPK, ok := pk.(*BN254ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu plonk: expected *BN254ProvingKey, got %T", pk)
		}
		return proveBN254(typedSPR, typedPK, fullWitness)
	default:
		return nil, fmt.Errorf("webgpu plonk: unsupported constraint system %T", spr)
	}
}

// PrepareWithCS initializes browser-side caches that need both the proving key
// and the constraint system. For PLONK this includes the static quotient
// numerator polynomials derived from the trace.
func PrepareWithCS(spr constraint.ConstraintSystem, pk gnarkplonk.ProvingKey) error {
	switch typedSPR := spr.(type) {
	case *csbn254.SparseR1CS:
		typedPK, ok := pk.(*BN254ProvingKey)
		if !ok {
			return fmt.Errorf("webgpu plonk: expected *BN254ProvingKey, got %T", pk)
		}
		return typedPK.prepareWithCS(typedSPR)
	default:
		return fmt.Errorf("webgpu plonk: unsupported constraint system %T", spr)
	}
}

// NewProvingKey returns an empty proving-key wrapper for supported curves.
func NewProvingKey(curveID ecc.ID) gnarkplonk.ProvingKey {
	switch curveID {
	case ecc.BN254:
		return &BN254ProvingKey{}
	default:
		panic("webgpu plonk: unsupported curve")
	}
}

// Prepare initializes browser-side caches for a deserialized proving key.
func Prepare(pk gnarkplonk.ProvingKey) error {
	switch typedPK := pk.(type) {
	case *BN254ProvingKey:
		return typedPK.ensurePrepared()
	default:
		return fmt.Errorf("webgpu plonk: unsupported proving key type %T", pk)
	}
}
