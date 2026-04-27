//go:build js && wasm

package groth16

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend"
	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	gnarkgroth16bls12377 "github.com/consensys/gnark/backend/groth16/bls12-377"
	gnarkgroth16bls12381 "github.com/consensys/gnark/backend/groth16/bls12-381"
	gnarkgroth16bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	csbls12377 "github.com/consensys/gnark/constraint/bls12-377"
	csbls12381 "github.com/consensys/gnark/constraint/bls12-381"
	csbn254 "github.com/consensys/gnark/constraint/bn254"
)

// Prove runs the Groth16 prover with browser/WebGPU acceleration on supported
// wasm targets.
//
// The current implementation accelerates the heavy quotient-H and MSM stages
// while leaving witness solving in Go. Commitment-carrying circuits are
// rejected for now.
func Prove(r1cs constraint.ConstraintSystem, pk gnarkgroth16.ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (gnarkgroth16.Proof, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		_tpk, ok := pk.(*BN254ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *BN254ProvingKey, got %T", pk)
		}
		return proveBN254(_r1cs, _tpk, fullWitness, opts...)
	case *csbls12377.R1CS:
		_tpk, ok := pk.(*BLS12377ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *BLS12377ProvingKey, got %T", pk)
		}
		return proveBLS12377(_r1cs, _tpk, fullWitness, opts...)
	case *csbls12381.R1CS:
		_tpk, ok := pk.(*BLS12381ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *BLS12381ProvingKey, got %T", pk)
		}
		return proveBLS12381(_r1cs, _tpk, fullWitness, opts...)
	default:
		return nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// Setup wraps gnark's Groth16 setup and returns a proving key wrapper that can
// later pin packed MSM bases in the browser runtime.
func Setup(r1cs constraint.ConstraintSystem) (gnarkgroth16.ProvingKey, gnarkgroth16.VerifyingKey, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		var pk BN254ProvingKey
		var vk gnarkgroth16bn254.VerifyingKey
		if err := gnarkgroth16bn254.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	case *csbls12377.R1CS:
		var pk BLS12377ProvingKey
		var vk gnarkgroth16bls12377.VerifyingKey
		if err := gnarkgroth16bls12377.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	case *csbls12381.R1CS:
		var pk BLS12381ProvingKey
		var vk gnarkgroth16bls12381.VerifyingKey
		if err := gnarkgroth16bls12381.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	default:
		return nil, nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// DummySetup mirrors gnark's DummySetup and returns a proving key wrapper for
// deserialization and testing.
func DummySetup(r1cs constraint.ConstraintSystem) (gnarkgroth16.ProvingKey, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		var pk BN254ProvingKey
		if err := gnarkgroth16bn254.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	case *csbls12377.R1CS:
		var pk BLS12377ProvingKey
		if err := gnarkgroth16bls12377.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	case *csbls12381.R1CS:
		var pk BLS12381ProvingKey
		if err := gnarkgroth16bls12381.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	default:
		return nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// NewProvingKey returns an empty proving key wrapper for supported curves.
func NewProvingKey(curveID ecc.ID) gnarkgroth16.ProvingKey {
	switch curveID {
	case ecc.BN254:
		return &BN254ProvingKey{}
	case ecc.BLS12_377:
		return &BLS12377ProvingKey{}
	case ecc.BLS12_381:
		return &BLS12381ProvingKey{}
	default:
		panic("webgpu groth16: unsupported curve")
	}
}

// Prepare initializes browser-side MSM caches for a deserialized proving key so
// the first proof does not include one-time bridge setup cost.
func Prepare(pk gnarkgroth16.ProvingKey) error {
	switch typed := pk.(type) {
	case *BN254ProvingKey:
		return typed.ensurePrepared()
	case *BLS12377ProvingKey:
		return typed.ensurePrepared()
	case *BLS12381ProvingKey:
		return typed.ensurePrepared()
	default:
		return fmt.Errorf("webgpu groth16: unsupported proving key type %T", pk)
	}
}
