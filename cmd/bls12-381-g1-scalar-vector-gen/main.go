package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ivokub/wnark-crypto/pkg/testgen"
)

func main() {
	out := flag.String("out", "testdata/vectors/g1/bls12_381_phase7_scalar_mul.json", "output vector path")
	flag.Parse()

	if err := run(*out); err != nil {
		panic(err)
	}
}

func run(outPath string) error {
	out := testgen.BuildBLS12381G1ScalarVectors()
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", outPath)
	return nil
}
