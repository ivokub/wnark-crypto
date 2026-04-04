package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

type Config struct {
	Curves []Curve `json:"curves"`
}

type Curve struct {
	ID      string      `json:"id"`
	GoConst string      `json:"goConst"`
	TSKey   string      `json:"tsKey"`
	Fields  []FieldSpec `json:"fields"`
}

type FieldSpec struct {
	ID         string `json:"id"`
	GoConst    string `json:"goConst"`
	TSKey      string `json:"tsKey"`
	StructName string `json:"structName"`
	HostWords  int    `json:"hostWords"`
	GPULimbs   int    `json:"gpuLimbs"`
	ByteSize   int    `json:"byteSize"`
}

//go:embed templates/curves_gen.go.tmpl
var goTemplate string

//go:embed templates/curve_shapes_gen.ts.tmpl
var tsTemplate string

//go:embed templates/field_types.wgsl.tmpl
var wgslTypeTemplate string

func main() {
	root, err := repoRoot()
	must(err)

	cfgBytes, err := os.ReadFile(filepath.Join(root, "configs", "curves.json"))
	must(err)

	var cfg Config
	must(json.Unmarshal(cfgBytes, &cfg))

	must(writeGo(root, cfg))
	must(writeTS(root, cfg))
	must(writeWGSLTypes(root, cfg))
}

func repoRoot() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(filepath.Join(wd, "go.mod")); err == nil {
		return wd, nil
	}
	root := filepath.Clean(filepath.Join(wd, "..", ".."))
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		return "", fmt.Errorf("resolve repo root: %w", err)
	}
	return root, nil
}

func writeGo(root string, cfg Config) error {
	tmpl, err := template.New("go").Parse(goTemplate)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, cfg); err != nil {
		return err
	}

	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		return fmt.Errorf("format generated go: %w", err)
	}

	path := filepath.Join(root, "go", "curvegpu", "curves_gen.go")
	return os.WriteFile(path, formatted, 0o644)
}

func writeTS(root string, cfg Config) error {
	tmpl, err := template.New("ts").Parse(tsTemplate)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, cfg); err != nil {
		return err
	}

	path := filepath.Join(root, "web", "src", "curvegpu", "curve_shapes_gen.ts")
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func writeWGSLTypes(root string, cfg Config) error {
	tmpl, err := template.New("wgsl").Parse(wgslTypeTemplate)
	if err != nil {
		return err
	}

	for _, curve := range cfg.Curves {
		for _, field := range curve.Fields {
			var buf bytes.Buffer
			data := struct {
				CurveID string
				Field   FieldSpec
			}{
				CurveID: curve.ID,
				Field:   field,
			}
			if err := tmpl.Execute(&buf, data); err != nil {
				return err
			}

			path := filepath.Join(root, "shaders", "curves", curve.ID, field.ID+"_types.wgsl")
			if err := os.WriteFile(path, bytes.TrimSpace(buf.Bytes()), 0o644); err != nil {
				return err
			}
			if err := appendTrailingNewline(path); err != nil {
				return err
			}
		}
	}
	return nil
}

func appendTrailingNewline(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if strings.HasSuffix(string(content), "\n") {
		return nil
	}
	return os.WriteFile(path, append(content, '\n'), 0o644)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
