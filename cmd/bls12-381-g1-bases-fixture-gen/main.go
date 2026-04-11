package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

const (
	fpBytes    = 48
	pointBytes = 144
)

type metadata struct {
	Count      int    `json:"count"`
	PointBytes int    `json:"point_bytes"`
	Format     string `json:"format"`
}

func main() {
	count := flag.Int("count", 1<<19, "number of affine multiples to generate")
	out := flag.String("out", "testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.bin", "output binary path")
	meta := flag.String("meta", "testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.json", "output metadata path")
	flag.Parse()

	if err := run(*count, *out, *meta); err != nil {
		panic(err)
	}
}

func run(count int, outPath string, metaPath string) error {
	if count <= 0 {
		return fmt.Errorf("count must be positive")
	}

	_, _, genAff, _ := gnarkbls12381.Generators()
	scalars := make([]gnarkfr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbls12381.BatchScalarMultiplicationG1(&genAff, scalars)
	oneMontZ := montOne()

	out := make([]byte, count*pointBytes)
	for i := range points {
		base := i * pointBytes
		writeElementLE(out[base:base+fpBytes], points[i].X)
		writeElementLE(out[base+fpBytes:base+2*fpBytes], points[i].Y)
		writeElementLE(out[base+2*fpBytes:base+3*fpBytes], oneMontZ)
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(outPath, out, 0o644); err != nil {
		return err
	}

	metaData := metadata{
		Count:      count,
		PointBytes: pointBytes,
		Format:     "jacobian_x_y_z_le",
	}
	metaJSON, err := json.MarshalIndent(metaData, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(metaPath, metaJSON, 0o644); err != nil {
		return err
	}

	fmt.Printf("wrote %s (%d points)\n", outPath, count)
	fmt.Printf("wrote %s\n", metaPath)
	return nil
}

func montOne() gnarkfp.Element {
	var one gnarkfp.Element
	one.SetOne()
	return one
}

func writeElementLE(dst []byte, v gnarkfp.Element) {
	for i, word := range [6]uint64(v) {
		base := i * 8
		dst[base+0] = byte(word)
		dst[base+1] = byte(word >> 8)
		dst[base+2] = byte(word >> 16)
		dst[base+3] = byte(word >> 24)
		dst[base+4] = byte(word >> 32)
		dst[base+5] = byte(word >> 40)
		dst[base+6] = byte(word >> 48)
		dst[base+7] = byte(word >> 56)
	}
}
