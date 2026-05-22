//go:build js && wasm

package main

import (
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	webgpuplonk "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/plonk"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/plonk/internal/wasmruntime"
)

func main() {
	if err := wasmruntime.Install(wasmruntime.Config{
		GlobalName: "wnarkPlonkRuntimeWebGPU",
		PKFactory:  webgpuplonk.NewProvingKey,
		Prepare:    webgpuplonk.Prepare,
		Prove: func(ccs constraint.ConstraintSystem, pk gnarkplonk.ProvingKey, fullWitness witness.Witness) (gnarkplonk.Proof, error) {
			return webgpuplonk.Prove(ccs, pk, fullWitness)
		},
	}); err != nil {
		panic(err)
	}
}
