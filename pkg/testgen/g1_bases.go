package testgen

import (
	"encoding/json"
	"math/rand"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	BN254FpBytes         = 32
	BN254PointBytes      = 96
	BN254G2PointBytes    = 192
	BLS12381FpBytes      = 48
	BLS12381PointBytes   = 144
	BLS12381G2PointBytes = 288
)

type BaseFixtureMetadata struct {
	Count      int    `json:"count"`
	PointBytes int    `json:"point_bytes"`
	Format     string `json:"format"`
}

func BuildRandomBLS12381G1Bases(count int, seed int64) ([]byte, error) {
	_, _, genAff, _ := gnarkbls12381.Generators()
	oneMontZ := montOneBLS12381()
	rng := rand.New(rand.NewSource(seed))
	scalars := make([]gnarkbls12381fr.Element, count)
	for i := range scalars {
		var raw [32]byte
		for j := range raw {
			raw[j] = byte(rng.Uint32())
		}
		scalars[i].SetBytes(raw[:])
		if scalars[i].IsZero() {
			scalars[i].SetUint64(1)
		}
	}
	points := gnarkbls12381.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BLS12381PointBytes)
	for i := range points {
		base := i * BLS12381PointBytes
		writeElementLE6(out[base:base+BLS12381FpBytes], points[i].X)
		writeElementLE6(out[base+BLS12381FpBytes:base+2*BLS12381FpBytes], points[i].Y)
		writeElementLE6(out[base+2*BLS12381FpBytes:base+3*BLS12381FpBytes], oneMontZ)
	}
	return out, nil
}

func BuildSequentialBLS12381G1Bases(count int) ([]byte, error) {
	_, _, genAff, _ := gnarkbls12381.Generators()
	oneMontZ := montOneBLS12381()
	scalars := make([]gnarkbls12381fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbls12381.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BLS12381PointBytes)
	for i := range points {
		base := i * BLS12381PointBytes
		writeElementLE6(out[base:base+BLS12381FpBytes], points[i].X)
		writeElementLE6(out[base+BLS12381FpBytes:base+2*BLS12381FpBytes], points[i].Y)
		writeElementLE6(out[base+2*BLS12381FpBytes:base+3*BLS12381FpBytes], oneMontZ)
	}
	return out, nil
}

func BuildSequentialBN254G1Bases(count int) ([]byte, error) {
	_, _, genAff, _ := gnarkbn254.Generators()
	oneMontZ := montOneBN254()
	scalars := make([]gnarkbn254fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbn254.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BN254PointBytes)
	for i := range points {
		base := i * BN254PointBytes
		writeElementLE4(out[base:base+BN254FpBytes], points[i].X)
		writeElementLE4(out[base+BN254FpBytes:base+2*BN254FpBytes], points[i].Y)
		writeElementLE4(out[base+2*BN254FpBytes:base+3*BN254FpBytes], oneMontZ)
	}
	return out, nil
}

func BuildSequentialBN254G2Bases(count int) ([]byte, error) {
	_, _, _, genAff := gnarkbn254.Generators()
	oneMontZ := montOneBN254()
	zero := gnarkbn254fp.Element{}
	scalars := make([]gnarkbn254fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbn254.BatchScalarMultiplicationG2(&genAff, scalars)

	out := make([]byte, count*BN254G2PointBytes)
	for i := range points {
		base := i * BN254G2PointBytes
		writeElementLE4(out[base:base+BN254FpBytes], points[i].X.A0)
		writeElementLE4(out[base+BN254FpBytes:base+2*BN254FpBytes], points[i].X.A1)
		writeElementLE4(out[base+2*BN254FpBytes:base+3*BN254FpBytes], points[i].Y.A0)
		writeElementLE4(out[base+3*BN254FpBytes:base+4*BN254FpBytes], points[i].Y.A1)
		writeElementLE4(out[base+4*BN254FpBytes:base+5*BN254FpBytes], oneMontZ)
		writeElementLE4(out[base+5*BN254FpBytes:base+6*BN254FpBytes], zero)
	}
	return out, nil
}

func BuildSequentialBLS12381G2Bases(count int) ([]byte, error) {
	_, _, _, genAff := gnarkbls12381.Generators()
	oneMontZ := montOneBLS12381()
	zero := gnarkbls12381fp.Element{}
	scalars := make([]gnarkbls12381fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbls12381.BatchScalarMultiplicationG2(&genAff, scalars)

	out := make([]byte, count*BLS12381G2PointBytes)
	for i := range points {
		base := i * BLS12381G2PointBytes
		writeElementLE6(out[base:base+BLS12381FpBytes], points[i].X.A0)
		writeElementLE6(out[base+BLS12381FpBytes:base+2*BLS12381FpBytes], points[i].X.A1)
		writeElementLE6(out[base+2*BLS12381FpBytes:base+3*BLS12381FpBytes], points[i].Y.A0)
		writeElementLE6(out[base+3*BLS12381FpBytes:base+4*BLS12381FpBytes], points[i].Y.A1)
		writeElementLE6(out[base+4*BLS12381FpBytes:base+5*BLS12381FpBytes], oneMontZ)
		writeElementLE6(out[base+5*BLS12381FpBytes:base+6*BLS12381FpBytes], zero)
	}
	return out, nil
}

func BuildBLS12381G1BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BLS12381PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func BuildBN254G1BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BN254PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func BuildBN254G2BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BN254G2PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func BuildBLS12381G2BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BLS12381G2PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func MarshalMetadataJSON(meta BaseFixtureMetadata) ([]byte, error) {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

func BuildRandomBN254G1Bases(count int, seed int64) ([]byte, error) {
	_, _, genAff, _ := gnarkbn254.Generators()
	oneMontZ := montOneBN254()
	rng := rand.New(rand.NewSource(seed))
	scalars := make([]gnarkbn254fr.Element, count)
	for i := range scalars {
		var raw [32]byte
		for j := range raw {
			raw[j] = byte(rng.Uint32())
		}
		scalars[i].SetBytes(raw[:])
		if scalars[i].IsZero() {
			scalars[i].SetUint64(1)
		}
	}
	points := gnarkbn254.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BN254PointBytes)
	for i := range points {
		base := i * BN254PointBytes
		writeElementLE4(out[base:base+BN254FpBytes], points[i].X)
		writeElementLE4(out[base+BN254FpBytes:base+2*BN254FpBytes], points[i].Y)
		writeElementLE4(out[base+2*BN254FpBytes:base+3*BN254FpBytes], oneMontZ)
	}
	return out, nil
}

func montOneBLS12381() gnarkbls12381fp.Element {
	var one gnarkbls12381fp.Element
	one.SetOne()
	return one
}

func montOneBN254() gnarkbn254fp.Element {
	var one gnarkbn254fp.Element
	one.SetOne()
	return one
}

func writeElementLE6(dst []byte, v gnarkbls12381fp.Element) {
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

func writeElementLE4(dst []byte, v gnarkbn254fp.Element) {
	for i, word := range [4]uint64(v) {
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
