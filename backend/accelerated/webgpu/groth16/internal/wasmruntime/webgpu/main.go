//go:build js && wasm

package main

import (
	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	webgpugroth16 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/internal/wasmruntime"
)

func main() {
	if err := wasmruntime.Install(wasmruntime.Config{
		GlobalName: "wnarkGroth16RuntimeWebGPU",
		PKFactory:  webgpugroth16.NewProvingKey,
		Prepare:    webgpugroth16.Prepare,
		Prove: func(ccs constraint.ConstraintSystem, pk gnarkgroth16.ProvingKey, fullWitness witness.Witness) (gnarkgroth16.Proof, error) {
			return webgpugroth16.Prove(ccs, pk, fullWitness)
		},
	}); err != nil {
		panic(err)
	}
}
