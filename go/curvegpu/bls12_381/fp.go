package bls12381

import (
	"encoding/binary"
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

type FpOp uint32

const (
	FpOpCopy FpOp = iota
	FpOpZero
	FpOpOne
	FpOpAdd
	FpOpSub
	FpOpNeg
	FpOpDouble
	FpOpNormalize
	FpOpEqual
	FpOpMul
	FpOpSquare
	FpOpToMont
	FpOpFromMont
)

const (
	fpElementBytes = 48
	fpUniformBytes = 16
	fpWorkgroup    = 64
)

type FpKernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
}

func ReadFpArithShader() (string, error) {
	return curvegpu.ReadShader("curves/bls12_381/fp_arith.wgsl")
}

func ZeroFpBatch(count int) []curvegpu.U32x12 {
	return make([]curvegpu.U32x12, count)
}

func NewFpKernel(device *wgpu.Device) (*FpKernel, error) {
	shaderText, err := ReadFpArithShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bls12-381-fp-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bls12-381-fp-bgl",
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
		Label:            "bls12-381-fp-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bls12-381-fp-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "fp_ops_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	return &FpKernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
	}, nil
}

func (k *FpKernel) Close() {
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

func RunFpOperation(device *wgpu.Device, op FpOp, a, b []curvegpu.U32x12) ([]curvegpu.U32x12, error) {
	kernel, err := NewFpKernel(device)
	if err != nil {
		return nil, err
	}
	defer kernel.Close()
	return kernel.Run(op, a, b)
}

func (k *FpKernel) Run(op FpOp, a, b []curvegpu.U32x12) ([]curvegpu.U32x12, error) {
	count, err := inferFpCount(a, b)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}

	inA := normalizeFpBatch(a, count)
	inB := normalizeFpBatch(b, count)

	inSize := uint64(count * fpElementBytes)
	outSize := inSize

	inputA, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fp-input-a",
		Size:  inSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_a buffer: %w", err)
	}
	defer inputA.Release()

	inputB, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fp-input-b",
		Size:  inSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_b buffer: %w", err)
	}
	defer inputB.Release()

	output, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fp-output",
		Size:  outSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("create output buffer: %w", err)
	}
	defer output.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fp-staging",
		Size:  outSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fp-params",
		Size:  fpUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	if err := k.device.Queue().WriteBuffer(inputA, 0, flattenFpBatch(inA)); err != nil {
		return nil, fmt.Errorf("write input_a buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(inputB, 0, flattenFpBatch(inB)); err != nil {
		return nil, fmt.Errorf("write input_b buffer: %w", err)
	}

	params := make([]byte, fpUniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(op))
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, fmt.Errorf("write uniform buffer: %w", err)
	}

	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bls12-381-fp-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputA, Size: inSize},
			{Binding: 1, Buffer: inputB, Size: inSize},
			{Binding: 2, Buffer: output, Size: outSize},
			{Binding: 3, Buffer: uniform, Size: fpUniformBytes},
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
	pass.Dispatch(uint32((count+fpWorkgroup-1)/fpWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(output, 0, staging, 0, outSize)

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

	data := make([]byte, outSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, fmt.Errorf("read output buffer: %w", err)
	}
	return unflattenFpBatch(data, count), nil
}

func inferFpCount(a, b []curvegpu.U32x12) (int, error) {
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

func normalizeFpBatch(in []curvegpu.U32x12, count int) []curvegpu.U32x12 {
	if len(in) == count {
		return in
	}
	out := make([]curvegpu.U32x12, count)
	copy(out, in)
	return out
}

func flattenFpBatch(in []curvegpu.U32x12) []byte {
	out := make([]byte, len(in)*fpElementBytes)
	for i, element := range in {
		base := i * fpElementBytes
		for j, limb := range element {
			binary.LittleEndian.PutUint32(out[base+j*4:], limb)
		}
	}
	return out
}

func unflattenFpBatch(data []byte, count int) []curvegpu.U32x12 {
	out := make([]curvegpu.U32x12, count)
	for i := 0; i < count; i++ {
		base := i * fpElementBytes
		for j := range out[i] {
			out[i][j] = binary.LittleEndian.Uint32(data[base+j*4:])
		}
	}
	return out
}
