package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"path/filepath"
	"strconv"
	"sync"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

const (
	fpBytes    = 48
	pointBytes = 144
)

type basesResponse struct {
	Count      int    `json:"count"`
	PointBytes int    `json:"point_bytes"`
	Format     string `json:"format"`
	Seed       int64  `json:"seed"`
}

type baseCache struct {
	mu      sync.RWMutex
	entries map[string][]byte
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8000", "listen address")
	root := flag.String("root", ".", "static file root")
	maxCount := flag.Int("max-count", 1<<20, "maximum supported base count")
	flag.Parse()

	cache := &baseCache{entries: make(map[string][]byte)}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/api/bls12-381/g1/bases.json", func(w http.ResponseWriter, r *http.Request) {
		count, seed, err := parseBaseRequest(r, *maxCount)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(basesResponse{
			Count:      count,
			PointBytes: pointBytes,
			Format:     "jacobian_x_y_z_le",
			Seed:       seed,
		})
	})
	mux.HandleFunc("/api/bls12-381/g1/bases.bin", func(w http.ResponseWriter, r *http.Request) {
		count, seed, err := parseBaseRequest(r, *maxCount)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		bytes, err := cache.getOrBuild(count, seed)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-Point-Bytes", strconv.Itoa(pointBytes))
		w.Header().Set("X-Base-Count", strconv.Itoa(count))
		w.Header().Set("X-Base-Seed", strconv.FormatInt(seed, 10))
		_, _ = w.Write(bytes)
	})

	fileServer := http.FileServer(http.Dir(filepath.Clean(*root)))
	mux.Handle("/", fileServer)

	log.Printf("curvegpu bench server listening on http://%s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

func parseBaseRequest(r *http.Request, maxCount int) (count int, seed int64, err error) {
	count = 1 << 19
	seed = 1
	if raw := r.URL.Query().Get("count"); raw != "" {
		count, err = strconv.Atoi(raw)
		if err != nil {
			return 0, 0, fmt.Errorf("invalid count: %w", err)
		}
	}
	if raw := r.URL.Query().Get("seed"); raw != "" {
		seed, err = strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return 0, 0, fmt.Errorf("invalid seed: %w", err)
		}
	}
	if count <= 0 {
		return 0, 0, fmt.Errorf("count must be positive")
	}
	if count > maxCount {
		return 0, 0, fmt.Errorf("count %d exceeds max-count %d", count, maxCount)
	}
	return count, seed, nil
}

func (c *baseCache) getOrBuild(count int, seed int64) ([]byte, error) {
	key := fmt.Sprintf("%d:%d", count, seed)
	c.mu.RLock()
	if cached, ok := c.entries[key]; ok {
		c.mu.RUnlock()
		return cached, nil
	}
	c.mu.RUnlock()

	bytes, err := buildRandomBases(count, seed)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.entries[key] = bytes
	c.mu.Unlock()
	return bytes, nil
}

func buildRandomBases(count int, seed int64) ([]byte, error) {
	_, _, genAff, _ := gnarkbls12381.Generators()
	oneMontZ := montOne()
	rng := rand.New(rand.NewSource(seed))
	scalars := make([]gnarkfr.Element, count)
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

	out := make([]byte, count*pointBytes)
	for i := range points {
		base := i * pointBytes
		writeElementLE(out[base:base+fpBytes], points[i].X)
		writeElementLE(out[base+fpBytes:base+2*fpBytes], points[i].Y)
		writeElementLE(out[base+2*fpBytes:base+3*fpBytes], oneMontZ)
	}
	return out, nil
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
