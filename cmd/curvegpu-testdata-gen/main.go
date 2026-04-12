package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

type target struct {
	name string
	args []string
}

var targets = []target{
	{name: "bn254-fr-vectors", args: []string{"go", "run", "./cmd/bn254-fr-vector-gen"}},
	{name: "bn254-fp-vectors", args: []string{"go", "run", "./cmd/bn254-fp-vector-gen"}},
	{name: "bn254-g1-vectors", args: []string{"go", "run", "./cmd/bn254-g1-vector-gen"}},
	{name: "bn254-g1-scalar-vectors", args: []string{"go", "run", "./cmd/bn254-g1-scalar-vector-gen"}},
	{name: "bn254-g1-msm-vectors", args: []string{"go", "run", "./cmd/bn254-g1-msm-vector-gen"}},
	{name: "bn254-fr-ntt-domains", args: []string{"go", "run", "./cmd/bn254-fr-ntt-domain-gen"}},
	{name: "bn254-fr-ntt-vectors", args: []string{"go", "run", "./cmd/bn254-fr-ntt-vector-gen"}},
	{name: "bls12-381-fr-vectors", args: []string{"go", "run", "./cmd/bls12-381-fr-vector-gen"}},
	{name: "bls12-381-fp-vectors", args: []string{"go", "run", "./cmd/bls12-381-fp-vector-gen"}},
	{name: "bls12-381-g1-vectors", args: []string{"go", "run", "./cmd/bls12-381-g1-vector-gen"}},
	{name: "bls12-381-g1-scalar-vectors", args: []string{"go", "run", "./cmd/bls12-381-g1-scalar-vector-gen"}},
	{name: "bls12-381-g1-msm-vectors", args: []string{"go", "run", "./cmd/bls12-381-g1-msm-vector-gen"}},
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
