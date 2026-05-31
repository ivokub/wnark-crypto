//go:build js && wasm

package groth16

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bls12377 "github.com/consensys/gnark/backend/groth16/bls12-377"
	groth16_bls12381 "github.com/consensys/gnark/backend/groth16/bls12-381"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	csbls12377 "github.com/consensys/gnark/constraint/bls12-377"
	csbls12381 "github.com/consensys/gnark/constraint/bls12-381"
	csbn254 "github.com/consensys/gnark/constraint/bn254"
	webgpu_bls12377 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/bls12-377"
	webgpu_bls12381 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/bls12-381"
	webgpu_bn254 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/bn254"
)

// Prove runs the Groth16 prover with browser/WebGPU acceleration on supported
// wasm targets.
//
// The current implementation accelerates the heavy quotient-H and MSM stages
// while leaving witness solving in Go. BSB22 commitments are handled by
// replacing gnark's commitment hint during solving and offloading the
// commitment/PoK MSMs to WebGPU.
func Prove(r1cs constraint.ConstraintSystem, pk groth16.ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (groth16.Proof, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		_tpk, ok := pk.(*webgpu_bn254.ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *webgpu_bn254.ProvingKey, got %T", pk)
		}
		return webgpu_bn254.Prove(_r1cs, _tpk, fullWitness, opts...)
	case *csbls12377.R1CS:
		_tpk, ok := pk.(*webgpu_bls12377.ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *webgpu_bls12377.ProvingKey, got %T", pk)
		}
		return webgpu_bls12377.Prove(_r1cs, _tpk, fullWitness, opts...)
	case *csbls12381.R1CS:
		_tpk, ok := pk.(*webgpu_bls12381.ProvingKey)
		if !ok {
			return nil, fmt.Errorf("webgpu groth16: expected *webgpu_bls12381.ProvingKey, got %T", pk)
		}
		return webgpu_bls12381.Prove(_r1cs, _tpk, fullWitness, opts...)
	default:
		return nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// Setup wraps gnark's Groth16 setup and returns a proving key wrapper that can
// later pin packed MSM bases in the browser runtime.
func Setup(r1cs constraint.ConstraintSystem) (groth16.ProvingKey, groth16.VerifyingKey, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		var pk webgpu_bn254.ProvingKey
		var vk groth16_bn254.VerifyingKey
		if err := groth16_bn254.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	case *csbls12377.R1CS:
		var pk webgpu_bls12377.ProvingKey
		var vk groth16_bls12377.VerifyingKey
		if err := groth16_bls12377.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	case *csbls12381.R1CS:
		var pk webgpu_bls12381.ProvingKey
		var vk groth16_bls12381.VerifyingKey
		if err := groth16_bls12381.Setup(_r1cs, &pk.ProvingKey, &vk); err != nil {
			return nil, nil, err
		}
		return &pk, &vk, nil
	default:
		return nil, nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// DummySetup mirrors gnark's DummySetup and returns a proving key wrapper for
// deserialization and testing.
func DummySetup(r1cs constraint.ConstraintSystem) (groth16.ProvingKey, error) {
	switch _r1cs := r1cs.(type) {
	case *csbn254.R1CS:
		var pk webgpu_bn254.ProvingKey
		if err := groth16_bn254.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	case *csbls12377.R1CS:
		var pk webgpu_bls12377.ProvingKey
		if err := groth16_bls12377.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	case *csbls12381.R1CS:
		var pk webgpu_bls12381.ProvingKey
		if err := groth16_bls12381.DummySetup(_r1cs, &pk.ProvingKey); err != nil {
			return nil, err
		}
		return &pk, nil
	default:
		return nil, fmt.Errorf("webgpu groth16: unsupported constraint system %T", r1cs)
	}
}

// NewProvingKey returns an empty proving key wrapper for supported curves.
func NewProvingKey(curveID ecc.ID) groth16.ProvingKey {
	switch curveID {
	case ecc.BN254:
		return &webgpu_bn254.ProvingKey{}
	case ecc.BLS12_377:
		return &webgpu_bls12377.ProvingKey{}
	case ecc.BLS12_381:
		return &webgpu_bls12381.ProvingKey{}
	default:
		panic("webgpu groth16: unsupported curve")
	}
}

// Prepare initializes browser-side MSM caches for a deserialized proving key so
// the first proof does not include one-time bridge setup cost.
func Prepare(pk groth16.ProvingKey) error {
	switch typed := pk.(type) {
	case *webgpu_bn254.ProvingKey:
		return typed.Prepare()
	case *webgpu_bls12377.ProvingKey:
		return typed.Prepare()
	case *webgpu_bls12381.ProvingKey:
		return typed.Prepare()
	default:
		return fmt.Errorf("webgpu groth16: unsupported proving key type %T", pk)
	}
}
