package main

import (
	"flag"
	"fmt"
	"os"

	blsfp "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bls12_381/fp"
	blsfr "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bls12_381/fr"
	blsg1 "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bls12_381/g1"
	blsg1msm "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bls12_381/g1_msm"
	blsg1scalar "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bls12_381/g1_scalar"
	bn254fp "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/fp"
	bn254fr "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/fr"
	bn254frntt "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/fr_ntt"
	bn254frvector "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/fr_vector"
	bn254g1 "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/g1"
	bn254g1msm "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/g1_msm"
	bn254g1scalar "github.com/ivokub/wnark-crypto/internal/metal/smoketest/bn254/g1_scalar"
)

func main() {
	var (
		curve = flag.String("curve", "bn254", "curve: bn254 | bls12_381")
		suite = flag.String("suite", "fr", "suite: fr | fp | fr_vector | fr_ntt | g1 | g1_scalar | g1_msm")
	)
	flag.Parse()

	if err := run(*curve, *suite); err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: %v\n", err)
		os.Exit(1)
	}
}

func run(curve, suite string) error {
	switch curve {
	case "bn254":
		switch suite {
		case "fr":
			return bn254fr.Run()
		case "fp":
			return bn254fp.Run()
		case "fr_vector":
			return bn254frvector.Run()
		case "fr_ntt":
			return bn254frntt.Run()
		case "g1":
			return bn254g1.Run()
		case "g1_scalar":
			return bn254g1scalar.Run()
		case "g1_msm":
			return bn254g1msm.Run()
		}
	case "bls12_381":
		switch suite {
		case "fr":
			return blsfr.Run()
		case "fp":
			return blsfp.Run()
		case "g1":
			return blsg1.Run()
		case "g1_scalar":
			return blsg1scalar.Run()
		case "g1_msm":
			return blsg1msm.Run()
		}
	}

	dieUnsupported(curve, suite)
	return nil
}

func dieUnsupported(curve, suite string) {
	fmt.Fprintf(os.Stderr, "unsupported metal smoke combination: curve=%s suite=%s\n", curve, suite)
	fmt.Fprintln(os.Stderr, "supported:")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fr")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fp")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fr_vector")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fr_ntt")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=g1")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=g1_scalar")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=g1_msm")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=fr")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=fp")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=g1")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=g1_scalar")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=g1_msm")
	os.Exit(2)
}
