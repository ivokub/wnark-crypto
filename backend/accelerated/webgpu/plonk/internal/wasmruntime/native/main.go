//go:build js && wasm

package main

import (
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/plonk/internal/wasmruntime"
)

func main() {
	if err := wasmruntime.Install(wasmruntime.Config{
		GlobalName: "wnarkPlonkRuntimeNative",
		PKFactory:  gnarkplonk.NewProvingKey,
		Prove: func(ccs constraint.ConstraintSystem, pk gnarkplonk.ProvingKey, fullWitness witness.Witness) (gnarkplonk.Proof, error) {
			return gnarkplonk.Prove(ccs, pk, fullWitness)
		},
	}); err != nil {
		panic(err)
	}
}
