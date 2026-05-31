//go:build js && wasm

package plonk

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	csbn254 "github.com/consensys/gnark/constraint/bn254"
	webgpu_bn254 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/plonk/bn254"
)

// Prove runs the PLONK prover for supported curves.
//
// For now only BN254 is supported. The implementation is a local copy of
// gnark's generated BN254 PLONK prover so we can replace individual prover
// phases with WebGPU calls without changing the browser-facing runtime API.
func Prove(spr constraint.ConstraintSystem, pk plonk.ProvingKey, fullWitness witness.Witness) (plonk.Proof, error) {
	switch typedSPR := spr.(type) {
	case *csbn254.SparseR1CS:
		typedPK, ok := pk.(*webgpu_bn254.ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu plonk: expected *webgpu_bn254.ProvingKey, got %T", pk)
		}
		return webgpu_bn254.Prove(typedSPR, typedPK, fullWitness)
	default:
		return nil, fmt.Errorf("webgpu plonk: unsupported constraint system %T", spr)
	}
}

// PrepareWithCS initializes browser-side caches that need both the proving key
// and the constraint system. For PLONK this includes the static quotient
// numerator polynomials derived from the trace.
func PrepareWithCS(spr constraint.ConstraintSystem, pk plonk.ProvingKey) error {
	switch typedSPR := spr.(type) {
	case *csbn254.SparseR1CS:
		typedPK, ok := pk.(*webgpu_bn254.ProvingKey)
		if !ok {
			return fmt.Errorf("webgpu plonk: expected *webgpu_bn254.ProvingKey, got %T", pk)
		}
		return typedPK.PrepareWithCS(typedSPR)
	default:
		return fmt.Errorf("webgpu plonk: unsupported constraint system %T", spr)
	}
}

// NewProvingKey returns an empty proving-key wrapper for supported curves.
func NewProvingKey(curveID ecc.ID) plonk.ProvingKey {
	switch curveID {
	case ecc.BN254:
		return &webgpu_bn254.ProvingKey{}
	default:
		panic("webgpu plonk: unsupported curve")
	}
}

// Prepare initializes browser-side caches for a deserialized proving key.
func Prepare(pk plonk.ProvingKey) error {
	switch typedPK := pk.(type) {
	case *webgpu_bn254.ProvingKey:
		return typedPK.Prepare()
	default:
		return fmt.Errorf("webgpu plonk: unsupported proving key type %T", pk)
	}
}
