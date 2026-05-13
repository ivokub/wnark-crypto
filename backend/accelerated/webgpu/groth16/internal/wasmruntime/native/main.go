//go:build js && wasm

package main

import (
	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/internal/wasmruntime"
)

func main() {
	if err := wasmruntime.Install(wasmruntime.Config{
		GlobalName: "wnarkGroth16RuntimeNative",
		PKFactory:  gnarkgroth16.NewProvingKey,
		Prove: func(ccs constraint.ConstraintSystem, pk gnarkgroth16.ProvingKey, fullWitness witness.Witness) (gnarkgroth16.Proof, error) {
			return gnarkgroth16.Prove(ccs, pk, fullWitness)
		},
	}); err != nil {
		panic(err)
	}
}
