package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/ivokub/wnark-crypto/pkg/testgen"
)

const (
	bn254PointBytes = testgen.BN254PointBytes
	pointBytes      = testgen.BLS12381PointBytes
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
	mux.HandleFunc("/api/bn254/g1/bases.json", func(w http.ResponseWriter, r *http.Request) {
		count, seed, err := parseBaseRequest(r, *maxCount)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(basesResponse{
			Count:      count,
			PointBytes: bn254PointBytes,
			Format:     "jacobian_x_y_z_le",
			Seed:       seed,
		})
	})
	mux.HandleFunc("/api/bn254/g1/bases.bin", func(w http.ResponseWriter, r *http.Request) {
		count, seed, err := parseBaseRequest(r, *maxCount)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		bytes, err := cache.getOrBuildBN254(count, seed)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-Point-Bytes", strconv.Itoa(bn254PointBytes))
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
	key := fmt.Sprintf("bls12-381:%d:%d", count, seed)
	c.mu.RLock()
	if cached, ok := c.entries[key]; ok {
		c.mu.RUnlock()
		return cached, nil
	}
	c.mu.RUnlock()

	bytes, err := testgen.BuildRandomBLS12381G1Bases(count, seed)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.entries[key] = bytes
	c.mu.Unlock()
	return bytes, nil
}

func (c *baseCache) getOrBuildBN254(count int, seed int64) ([]byte, error) {
	key := fmt.Sprintf("bn254:%d:%d", count, seed)
	c.mu.RLock()
	if cached, ok := c.entries[key]; ok {
		c.mu.RUnlock()
		return cached, nil
	}
	c.mu.RUnlock()

	bytes, err := testgen.BuildRandomBN254G1Bases(count, seed)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.entries[key] = bytes
	c.mu.Unlock()
	return bytes, nil
}
