package main

import (
	"fmt"
	"log"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

func main() {
	shaders, err := curvegpu.ListShaders()
	if err != nil {
		log.Fatalf("list shaders: %v", err)
	}

	bn254FR, err := curvegpu.ShapeFor(curvegpu.CurveBN254, curvegpu.FieldScalar)
	if err != nil {
		log.Fatalf("shape lookup: %v", err)
	}

	fmt.Printf("curvegpu smoke: %d shaders discovered\n", len(shaders))
	fmt.Printf("bn254 fr hostWords=%d gpuLimbs=%d byteSize=%d\n", bn254FR.HostWords, bn254FR.GPULimbs, bn254FR.ByteSize)
}
