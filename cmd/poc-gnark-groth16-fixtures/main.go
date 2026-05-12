package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/ivokub/wnark-crypto/poc-gnark-groth16/common"
)

func main() {
	var curveName string
	var logsCSV string
	var commitmentsCSV string
	var outDir string

	flag.StringVar(&curveName, "curve", "all", "curve to generate: bn254, bls12_377, bls12_381, or all")
	flag.StringVar(&logsCSV, "logs", "12,15,18", "comma-separated circuit size logs")
	flag.StringVar(&commitmentsCSV, "commitments", "0,1,2", "comma-separated commitment counts")
	flag.StringVar(&outDir, "out", "poc-gnark-groth16/fixtures", "output fixture root")
	flag.Parse()

	logs, err := parseLogs(logsCSV)
	if err != nil {
		exit(err)
	}
	commitmentCounts, err := parseCommitments(commitmentsCSV)
	if err != nil {
		exit(err)
	}

	curves, err := selectCurves(curveName)
	if err != nil {
		exit(err)
	}

	for _, curveID := range curves {
		name := curveKey(curveID)
		for _, sizeLog := range logs {
			for _, commitments := range commitmentCounts {
				if err := generateFixture(curveID, name, sizeLog, commitments, outDir); err != nil {
					exit(err)
				}
			}
		}
	}
}

func generateFixture(curveID ecc.ID, curve string, sizeLog, commitments int, outDir string) error {
	depth := 1 << sizeLog
	circuit := &common.MulAddChainCircuit{Steps: depth, Commitments: commitments}
	ccs, err := frontend.Compile(curveID.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		return fmt.Errorf("compile %s 2^%d commit%d: %w", curve, sizeLog, commitments, err)
	}

	pk, vk, err := gnarkgroth16.Setup(ccs)
	if err != nil {
		return fmt.Errorf("setup %s 2^%d commit%d: %w", curve, sizeLog, commitments, err)
	}

	base := filepath.Join(outDir, curve, fmt.Sprintf("2pow%d", sizeLog), fmt.Sprintf("commit%d", commitments))
	if err := os.MkdirAll(base, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", base, err)
	}

	if err := writeBinary(filepath.Join(base, "ccs.bin"), ccs); err != nil {
		return err
	}
	if err := writeDump(filepath.Join(base, "pk.dump"), pk); err != nil {
		return err
	}
	if err := writeBinary(filepath.Join(base, "vk.bin"), vk); err != nil {
		return err
	}

	fmt.Printf("wrote %s 2^%d commit%d fixtures under %s\n", curve, sizeLog, commitments, base)
	return nil
}

func writeBinary(path string, value io.WriterTo) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	if _, err := value.WriteTo(f); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

func writeDump(path string, value interface{ WriteDump(io.Writer) error }) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	if err := value.WriteDump(f); err != nil {
		return fmt.Errorf("write dump %s: %w", path, err)
	}
	return nil
}

func parseLogs(csv string) ([]int, error) {
	parts := strings.Split(csv, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid log %q", part)
		}
		if value <= 0 {
			return nil, fmt.Errorf("invalid log %d", value)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no logs provided")
	}
	return out, nil
}

func parseCommitments(csv string) ([]int, error) {
	parts := strings.Split(csv, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid commitments %q", part)
		}
		if value < 0 || value > 2 {
			return nil, fmt.Errorf("invalid commitments %d", value)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no commitment counts provided")
	}
	return out, nil
}

func selectCurves(curveName string) ([]ecc.ID, error) {
	switch curveName {
	case "all":
		return []ecc.ID{ecc.BN254, ecc.BLS12_377, ecc.BLS12_381}, nil
	case "bn254":
		return []ecc.ID{ecc.BN254}, nil
	case "bls12_377":
		return []ecc.ID{ecc.BLS12_377}, nil
	case "bls12_381":
		return []ecc.ID{ecc.BLS12_381}, nil
	default:
		return nil, fmt.Errorf("unsupported curve %q", curveName)
	}
}

func curveKey(curveID ecc.ID) string {
	switch curveID {
	case ecc.BN254:
		return "bn254"
	case ecc.BLS12_377:
		return "bls12_377"
	case ecc.BLS12_381:
		return "bls12_381"
	default:
		panic("unsupported curve")
	}
}

func exit(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
