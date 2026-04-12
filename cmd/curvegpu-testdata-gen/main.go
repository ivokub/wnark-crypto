package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/ivokub/wnark-crypto/pkg/testgen"
)

type target struct {
	name string
	args []string
	run  func(string) error
}

var targets = []target{
	{name: "bn254-fr-vectors", args: []string{"go", "run", "./cmd/bn254-fr-vector-gen"}},
	{name: "bn254-fp-vectors", args: []string{"go", "run", "./cmd/bn254-fp-vector-gen"}},
	{name: "bn254-g1-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_phase6_g1_ops.json"), testgen.BuildBN254G1OpsVectors())
	}},
	{name: "bn254-g1-scalar-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_phase7_scalar_mul.json"), testgen.BuildBN254G1ScalarVectors())
	}},
	{name: "bn254-g1-msm-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_phase8_msm.json"), testgen.BuildBN254G1MSMVectors())
	}},
	{name: "bn254-fr-ntt-domains", args: []string{"go", "run", "./cmd/bn254-fr-ntt-domain-gen"}},
	{name: "bn254-fr-ntt-vectors", args: []string{"go", "run", "./cmd/bn254-fr-ntt-vector-gen"}},
	{name: "bls12-381-fr-vectors", args: []string{"go", "run", "./cmd/bls12-381-fr-vector-gen"}},
	{name: "bls12-381-fp-vectors", args: []string{"go", "run", "./cmd/bls12-381-fp-vector-gen"}},
	{name: "bls12-381-g1-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_phase6_g1_ops.json"), testgen.BuildBLS12381G1OpsVectors())
	}},
	{name: "bls12-381-g1-scalar-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_phase7_scalar_mul.json"), testgen.BuildBLS12381G1ScalarVectors())
	}},
	{name: "bls12-381-g1-msm-vectors", run: func(root string) error {
		return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_phase8_msm.json"), testgen.BuildBLS12381G1MSMVectors())
	}},
	{name: "bls12-381-g1-bases-fixture", args: []string{"go", "run", "./cmd/bls12-381-g1-bases-fixture-gen"}},
}

func main() {
	list := flag.Bool("list", false, "list available targets")
	only := flag.String("target", "", "run only a single target name")
	flag.Parse()

	if *list {
		for _, t := range targets {
			fmt.Println(t.name)
		}
		return
	}

	repoRoot, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	repoRoot, err = filepath.Abs(repoRoot)
	if err != nil {
		panic(err)
	}

	selected := targets
	if *only != "" {
		selected = nil
		for _, t := range targets {
			if t.name == *only {
				selected = append(selected, t)
				break
			}
		}
		if len(selected) == 0 {
			panic(fmt.Errorf("unknown target: %s", *only))
		}
	}

	for _, t := range selected {
		fmt.Printf("running %s\n", t.name)
		if t.run != nil {
			if err := t.run(repoRoot); err != nil {
				panic(fmt.Errorf("%s failed: %w", t.name, err))
			}
			continue
		}
		cmd := exec.Command(t.args[0], t.args[1:]...)
		cmd.Dir = repoRoot
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = os.Environ()
		if err := cmd.Run(); err != nil {
			panic(fmt.Errorf("%s failed: %w", t.name, err))
		}
	}
}

func writeJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", path)
	return nil
}
