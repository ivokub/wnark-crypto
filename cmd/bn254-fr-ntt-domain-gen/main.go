package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"path/filepath"

	gnarkfft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

type domainMetaFile struct {
	Domains []domainMeta `json:"domains"`
}

type domainMeta struct {
	LogN             int    `json:"log_n"`
	Size             int    `json:"size"`
	OmegaHex         string `json:"omega_hex"`
	OmegaInvHex      string `json:"omega_inv_hex"`
	CardinalityInvHex string `json:"cardinality_inv_hex"`
}

func main() {
	var (
		minLog = flag.Int("min-log", 10, "minimum log2 size")
		maxLog = flag.Int("max-log", 20, "maximum log2 size")
		out    = flag.String("out", filepath.FromSlash("testdata/vectors/fr/bn254_ntt_domains.json"), "output path")
	)
	flag.Parse()

	if *minLog < 1 || *maxLog < *minLog {
		panic("invalid min-log/max-log")
	}

	outFile := domainMetaFile{
		Domains: make([]domainMeta, 0, *maxLog-*minLog+1),
	}
	for logN := *minLog; logN <= *maxLog; logN++ {
		size := 1 << logN
		domain := gnarkfft.NewDomain(uint64(size))
		outFile.Domains = append(outFile.Domains, domainMeta{
			LogN:             logN,
			Size:             size,
			OmegaHex:         domain.Generator.BigInt(new(big.Int)).Text(16),
			OmegaInvHex:      domain.GeneratorInv.BigInt(new(big.Int)).Text(16),
			CardinalityInvHex: domain.CardinalityInv.BigInt(new(big.Int)).Text(16),
		})
	}

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		panic(err)
	}
	data, err := json.MarshalIndent(outFile, "", "  ")
	if err != nil {
		panic(err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(*out, data, 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s\n", *out)
}
