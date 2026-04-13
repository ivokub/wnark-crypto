package bn254

import (
	"encoding/binary"
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/internal/metal"
)

type G1Op uint32

const (
	G1OpCopy G1Op = iota
	G1OpJacInfinity
	G1OpAffineToJac
	G1OpNegJac
	G1OpDoubleJac
	G1OpAddMixed
	G1OpJacToAffine
	G1OpAffineAdd
	G1OpScalarMulAffine
)

const (
	g1PointBytes   = 96
	g1UniformBytes = 16
	g1Workgroup    = 64
)

var bn254FpOneMont = metal.U32x8{
	0xc58f0d9d,
	0xd35d438d,
	0xf5c70b3d,
	0x0a78eb28,
	0x7879462c,
	0x666ea36f,
	0x9a07df2f,
	0x0e0a77c1,
}

type G1Affine struct {
	X metal.U32x8
	Y metal.U32x8
}

type G1Jac struct {
	X metal.U32x8
	Y metal.U32x8
	Z metal.U32x8
}

type G1Kernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
}

func ReadG1ArithShader() (string, error) {
	return metal.ReadShaderParts(
		"curves/bn254/fp_arith.wgsl#section=fp-types",
		"curves/bn254/fp_arith.wgsl#section=fp-consts",
		"curves/bn254/fp_arith.wgsl#section=fp-core",
		"curves/bn254/g1_arith.wgsl",
	)
}

func NewG1Kernel(device *wgpu.Device) (*G1Kernel, error) {
	shaderText, err := ReadG1ArithShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bn254-g1-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bn254-g1-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		shader.Release()
		return nil, fmt.Errorf("create bind group layout: %w", err)
	}
	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "bn254-g1-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bn254-g1-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "g1_ops_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	return &G1Kernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
	}, nil
}

func (k *G1Kernel) Close() {
	if k == nil {
		return
	}
	if k.pipeline != nil {
		k.pipeline.Release()
	}
	if k.pipelineLayout != nil {
		k.pipelineLayout.Release()
	}
	if k.layout != nil {
		k.layout.Release()
	}
	if k.shader != nil {
		k.shader.Release()
	}
}

func G1JacInfinity() G1Jac {
	return G1Jac{
		X: bn254FpOneMont,
		Y: bn254FpOneMont,
	}
}

func ZeroG1Batch(count int) []G1Jac {
	return make([]G1Jac, count)
}

func (p G1Affine) IsInfinity() bool {
	var zero metal.U32x8
	return p.X == zero && p.Y == zero
}

func (p G1Affine) AsKernelInput() G1Jac {
	if p.IsInfinity() {
		return G1Jac{}
	}
	return G1Jac{
		X: p.X,
		Y: p.Y,
		Z: bn254FpOneMont,
	}
}

func (p G1Jac) IsInfinity() bool {
	var zero metal.U32x8
	return p.Z == zero
}

func AffineToKernelBatch(in []G1Affine) []G1Jac {
	out := make([]G1Jac, len(in))
	for i := range in {
		out[i] = in[i].AsKernelInput()
	}
	return out
}

func (k *G1Kernel) Run(op G1Op, a, b []G1Jac) ([]G1Jac, error) {
	count, err := inferG1Count(a, b)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}

	inA := normalizeG1Batch(a, count)
	inB := normalizeG1Batch(b, count)
	bufSize := uint64(count * g1PointBytes)

	inputA, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-g1-input-a",
		Size:  bufSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_a buffer: %w", err)
	}
	defer inputA.Release()

	inputB, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-g1-input-b",
		Size:  bufSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_b buffer: %w", err)
	}
	defer inputB.Release()

	output, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-g1-output",
		Size:  bufSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("create output buffer: %w", err)
	}
	defer output.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-g1-staging",
		Size:  bufSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-g1-params",
		Size:  g1UniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	if err := k.device.Queue().WriteBuffer(inputA, 0, flattenG1Batch(inA)); err != nil {
		return nil, fmt.Errorf("write input_a buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(inputB, 0, flattenG1Batch(inB)); err != nil {
		return nil, fmt.Errorf("write input_b buffer: %w", err)
	}

	params := make([]byte, g1UniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(op))
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, fmt.Errorf("write uniform buffer: %w", err)
	}

	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bn254-g1-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputA, Size: bufSize},
			{Binding: 1, Buffer: inputB, Size: bufSize},
			{Binding: 2, Buffer: output, Size: bufSize},
			{Binding: 3, Buffer: uniform, Size: g1UniformBytes},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create bind group: %w", err)
	}
	defer bindGroup.Release()

	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("create command encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return nil, fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32((count+g1Workgroup-1)/g1Workgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(output, 0, staging, 0, bufSize)

	cmd, err := encoder.Finish()
	if err != nil {
		return nil, fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, fmt.Errorf("submit: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, fmt.Errorf("wait idle: %w", err)
	}

	data := make([]byte, bufSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, fmt.Errorf("read output buffer: %w", err)
	}
	return unflattenG1Batch(data, count), nil
}

func (k *G1Kernel) Copy(a []G1Jac) ([]G1Jac, error) {
	return k.Run(G1OpCopy, a, ZeroG1Batch(len(a)))
}

func (k *G1Kernel) Infinity(count int) ([]G1Jac, error) {
	return k.Run(G1OpJacInfinity, ZeroG1Batch(count), ZeroG1Batch(count))
}

func (k *G1Kernel) AffineToJac(a []G1Affine) ([]G1Jac, error) {
	return k.Run(G1OpAffineToJac, AffineToKernelBatch(a), ZeroG1Batch(len(a)))
}

func (k *G1Kernel) NegJac(a []G1Jac) ([]G1Jac, error) {
	return k.Run(G1OpNegJac, a, ZeroG1Batch(len(a)))
}

func (k *G1Kernel) DoubleJac(a []G1Jac) ([]G1Jac, error) {
	return k.Run(G1OpDoubleJac, a, ZeroG1Batch(len(a)))
}

func (k *G1Kernel) AddMixed(p []G1Jac, q []G1Affine) ([]G1Jac, error) {
	return k.Run(G1OpAddMixed, p, AffineToKernelBatch(q))
}

func (k *G1Kernel) JacToAffine(p []G1Jac) ([]G1Jac, error) {
	return k.Run(G1OpJacToAffine, p, ZeroG1Batch(len(p)))
}

func (k *G1Kernel) AffineAdd(a, b []G1Affine) ([]G1Jac, error) {
	return k.Run(G1OpAffineAdd, AffineToKernelBatch(a), AffineToKernelBatch(b))
}

func (k *G1Kernel) ScalarMulAffine(base []G1Affine, scalars []metal.U32x8) ([]G1Jac, error) {
	return k.Run(G1OpScalarMulAffine, AffineToKernelBatch(base), scalarWordsToBatch(scalars))
}

func inferG1Count(a, b []G1Jac) (int, error) {
	switch {
	case len(a) == 0 && len(b) == 0:
		return 0, nil
	case len(a) == 0:
		return len(b), nil
	case len(b) == 0:
		return len(a), nil
	case len(a) != len(b):
		return 0, fmt.Errorf("input length mismatch: %d != %d", len(a), len(b))
	default:
		return len(a), nil
	}
}

func normalizeG1Batch(in []G1Jac, count int) []G1Jac {
	if len(in) == count {
		return in
	}
	out := make([]G1Jac, count)
	copy(out, in)
	return out
}

func flattenG1Batch(in []G1Jac) []byte {
	out := make([]byte, len(in)*g1PointBytes)
	for i, point := range in {
		base := i * g1PointBytes
		for j, limb := range point.X {
			binary.LittleEndian.PutUint32(out[base+j*4:], limb)
		}
		for j, limb := range point.Y {
			binary.LittleEndian.PutUint32(out[base+32+j*4:], limb)
		}
		for j, limb := range point.Z {
			binary.LittleEndian.PutUint32(out[base+64+j*4:], limb)
		}
	}
	return out
}

func unflattenG1Batch(data []byte, count int) []G1Jac {
	out := make([]G1Jac, count)
	for i := 0; i < count; i++ {
		base := i * g1PointBytes
		for j := range out[i].X {
			out[i].X[j] = binary.LittleEndian.Uint32(data[base+j*4:])
			out[i].Y[j] = binary.LittleEndian.Uint32(data[base+32+j*4:])
			out[i].Z[j] = binary.LittleEndian.Uint32(data[base+64+j*4:])
		}
	}
	return out
}

func scalarWordsToBatch(in []metal.U32x8) []G1Jac {
	out := make([]G1Jac, len(in))
	for i := range in {
		out[i].X = in[i]
	}
	return out
}
