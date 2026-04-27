package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ivokub/wnark-crypto/pkg/testgen"
)

type target struct {
	name           string
	defaultEnabled bool
	run            func(string) error
}

func buildTargets(g1FixtureCount int, g2FixtureCount int) []target {
	return []target{
		{name: "bn254-fr-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bn254_fr_ops.json"), testgen.BuildBN254FROpsVectors())
		}},
		{name: "bn254-fp-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fp/bn254_fp_ops.json"), testgen.BuildBN254FPOpsVectors())
		}},
		{name: "bn254-fr-vector-ops", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bn254_fr_vector_ops.json"), testgen.BuildBN254FRVectorOps())
		}},
		{name: "bn254-g1-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_g1_ops.json"), testgen.BuildBN254G1OpsVectors())
		}},
		{name: "bn254-g2-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bn254_g2_ops.json"), testgen.BuildBN254G2OpsVectors())
		}},
		{name: "bn254-g1-scalar-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_g1_scalar_mul.json"), testgen.BuildBN254G1ScalarVectors())
		}},
		{name: "bn254-g1-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bn254_g1_msm.json"), testgen.BuildBN254G1MSMVectors())
		}},
		{name: "bn254-g2-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bn254_g2_msm.json"), testgen.BuildBN254G2MSMVectors())
		}},
		{name: "bn254-fr-ntt-domains", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bn254_ntt_domains.json"), testgen.BuildBN254NTTDomainFile(3, 20))
		}},
		{name: "bn254-fr-ntt-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bn254_fr_ntt.json"), testgen.BuildBN254NTTVectors())
		}},
		{name: "bls12-381-fr-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_381_fr_ops.json"), testgen.BuildBLS12381FROpsVectors())
		}},
		{name: "bls12-377-fr-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_377_fr_ops.json"), testgen.BuildBLS12377FROpsVectors())
		}},
		{name: "bls12-381-fr-vector-ops", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_381_fr_vector_ops.json"), testgen.BuildBLS12381FRVectorOps())
		}},
		{name: "bls12-377-fr-vector-ops", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_377_fr_vector_ops.json"), testgen.BuildBLS12377FRVectorOps())
		}},
		{name: "bls12-381-fr-ntt-domains", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_381_ntt_domains.json"), testgen.BuildBLS12381NTTDomainFile(3, 20))
		}},
		{name: "bls12-377-fr-ntt-domains", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_377_ntt_domains.json"), testgen.BuildBLS12377NTTDomainFile(3, 20))
		}},
		{name: "bls12-381-fr-ntt-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_381_fr_ntt.json"), testgen.BuildBLS12381NTTVectors())
		}},
		{name: "bls12-377-fr-ntt-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fr/bls12_377_fr_ntt.json"), testgen.BuildBLS12377NTTVectors())
		}},
		{name: "bls12-381-fp-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fp/bls12_381_fp_ops.json"), testgen.BuildBLS12381FPOpsVectors())
		}},
		{name: "bls12-377-fp-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/fp/bls12_377_fp_ops.json"), testgen.BuildBLS12377FPOpsVectors())
		}},
		{name: "bls12-381-g1-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_g1_ops.json"), testgen.BuildBLS12381G1OpsVectors())
		}},
		{name: "bls12-377-g1-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_377_g1_ops.json"), testgen.BuildBLS12377G1OpsVectors())
		}},
		{name: "bls12-381-g2-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bls12_381_g2_ops.json"), testgen.BuildBLS12381G2OpsVectors())
		}},
		{name: "bls12-377-g2-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bls12_377_g2_ops.json"), testgen.BuildBLS12377G2OpsVectors())
		}},
		{name: "bls12-381-g1-scalar-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_g1_scalar_mul.json"), testgen.BuildBLS12381G1ScalarVectors())
		}},
		{name: "bls12-377-g1-scalar-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_377_g1_scalar_mul.json"), testgen.BuildBLS12377G1ScalarVectors())
		}},
		{name: "bls12-381-g1-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_381_g1_msm.json"), testgen.BuildBLS12381G1MSMVectors())
		}},
		{name: "bls12-377-g1-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g1/bls12_377_g1_msm.json"), testgen.BuildBLS12377G1MSMVectors())
		}},
		{name: "bls12-381-g2-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bls12_381_g2_msm.json"), testgen.BuildBLS12381G2MSMVectors())
		}},
		{name: "bls12-377-g2-msm-vectors", defaultEnabled: true, run: func(root string) error {
			return writeJSON(filepath.Join(root, "testdata/vectors/g2/bls12_377_g2_msm.json"), testgen.BuildBLS12377G2MSMVectors())
		}},
		{name: "bn254-g1-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g1/bn254_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g1/bn254_bases_jacobian.json")
			data, err := testgen.BuildSequentialBN254G1Bases(g1FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBN254G1BaseFixtureMetadata(g1FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g1FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
		{name: "bls12-381-g1-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g1/bls12_381_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g1/bls12_381_bases_jacobian.json")
			data, err := testgen.BuildSequentialBLS12381G1Bases(g1FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBLS12381G1BaseFixtureMetadata(g1FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g1FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
		{name: "bls12-377-g1-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g1/bls12_377_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g1/bls12_377_bases_jacobian.json")
			data, err := testgen.BuildSequentialBLS12377G1Bases(g1FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBLS12377G1BaseFixtureMetadata(g1FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g1FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
		{name: "bn254-g2-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g2/bn254_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g2/bn254_bases_jacobian.json")
			data, err := testgen.BuildSequentialBN254G2Bases(g2FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBN254G2BaseFixtureMetadata(g2FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g2FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
		{name: "bls12-381-g2-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g2/bls12_381_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g2/bls12_381_bases_jacobian.json")
			data, err := testgen.BuildSequentialBLS12381G2Bases(g2FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBLS12381G2BaseFixtureMetadata(g2FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g2FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
		{name: "bls12-377-g2-bases-fixture", defaultEnabled: true, run: func(root string) error {
			outPath := filepath.Join(root, "testdata/fixtures/g2/bls12_377_bases_jacobian.bin")
			metaPath := filepath.Join(root, "testdata/fixtures/g2/bls12_377_bases_jacobian.json")
			data, err := testgen.BuildSequentialBLS12377G2Bases(g2FixtureCount)
			if err != nil {
				return err
			}
			if err := writeBinary(outPath, data); err != nil {
				return err
			}
			metaJSON, err := testgen.MarshalMetadataJSON(testgen.BuildBLS12377G2BaseFixtureMetadata(g2FixtureCount))
			if err != nil {
				return err
			}
			if err := writeBinary(metaPath, metaJSON); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d points)\n", outPath, g2FixtureCount)
			fmt.Printf("wrote %s\n", metaPath)
			return nil
		}},
	}
}

func main() {
	list := flag.Bool("list", false, "list available targets")
	only := flag.String("target", "", "run only a single target name")
	fixtureCount := flag.Int("fixture-count", 1<<15, "point count for G1 base fixture targets")
	g2FixtureCount := flag.Int("g2-fixture-count", 1<<14, "point count for G2 base fixture targets")
	flag.Parse()

	targets := buildTargets(*fixtureCount, *g2FixtureCount)

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
	} else {
		selected = nil
		for _, t := range targets {
			if t.defaultEnabled {
				selected = append(selected, t)
			}
		}
	}

	for _, t := range selected {
		fmt.Printf("running %s\n", t.name)
		if err := t.run(repoRoot); err != nil {
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

func writeBinary(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
