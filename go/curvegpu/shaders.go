package curvegpu

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

const shaderSectionPrefix = "#section="

func repoRoot() (string, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime caller lookup failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..")), nil
}

func ShaderRoot() (string, error) {
	root, err := repoRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "shaders"), nil
}

func ReadShader(relPath string) (string, error) {
	root, err := ShaderRoot()
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(relPath)))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func ReadShaderParts(parts ...string) (string, error) {
	if len(parts) == 0 {
		return "", fmt.Errorf("no shader parts requested")
	}
	var b strings.Builder
	for _, part := range parts {
		text, err := ReadShaderPart(part)
		if err != nil {
			return "", err
		}
		b.WriteString(text)
		if !strings.HasSuffix(text, "\n") {
			b.WriteByte('\n')
		}
	}
	return b.String(), nil
}

func ReadShaderPart(spec string) (string, error) {
	path, section, hasSection := strings.Cut(spec, shaderSectionPrefix)
	text, err := ReadShader(path)
	if err != nil {
		return "", err
	}
	if !hasSection {
		return text, nil
	}
	return extractShaderSection(text, section)
}

func extractShaderSection(text, section string) (string, error) {
	begin := "// curvegpu:section " + section + " begin"
	end := "// curvegpu:section " + section + " end"
	start := strings.Index(text, begin)
	if start < 0 {
		return "", fmt.Errorf("shader section %q begin marker not found", section)
	}
	start += len(begin)
	if start < len(text) && text[start] == '\r' {
		start++
	}
	if start < len(text) && text[start] == '\n' {
		start++
	}
	stop := strings.Index(text[start:], end)
	if stop < 0 {
		return "", fmt.Errorf("shader section %q end marker not found", section)
	}
	return text[start : start+stop], nil
}

func ListShaders() ([]string, error) {
	root, err := ShaderRoot()
	if err != nil {
		return nil, err
	}
	var out []string
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if !strings.HasSuffix(path, ".wgsl") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		out = append(out, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}
