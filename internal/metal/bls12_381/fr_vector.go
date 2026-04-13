package bls12381

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/internal/metal"
)

type FrVectorOp uint32

const (
	FrVectorOpCopy FrVectorOp = iota
	FrVectorOpAdd
	FrVectorOpSub
	FrVectorOpMulFactors
	FrVectorOpBitReverseCopy
)

const frVectorUniformBytes = 16

type FrVectorKernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
}

type FrVectorProfile struct {
	Upload   time.Duration
	Kernel   time.Duration
	Readback time.Duration
	Total    time.Duration
}

func ReadFrVectorShader() (string, error) {
	return metal.ReadShader("curves/bls12_381/fr_vector.wgsl")
}

func NewFrVectorKernel(device *wgpu.Device) (*FrVectorKernel, error) {
	shaderText, err := ReadFrVectorShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bls12-381-fr-vector-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bls12-381-fr-vector-bgl",
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
		Label:            "bls12-381-fr-vector-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bls12-381-fr-vector-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "fr_vector_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	return &FrVectorKernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
	}, nil
}

func (k *FrVectorKernel) Close() {
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

func (k *FrVectorKernel) Run(op FrVectorOp, input, aux []metal.U32x8, logCount uint32) ([]metal.U32x8, error) {
	out, _, err := k.RunProfiled(op, input, aux, logCount)
	return out, err
}

func (k *FrVectorKernel) RunProfiled(op FrVectorOp, input, aux []metal.U32x8, logCount uint32) ([]metal.U32x8, FrVectorProfile, error) {
	count := len(input)
	if count == 0 {
		return nil, FrVectorProfile{}, nil
	}
	if len(aux) != 0 && len(aux) != count {
		return nil, FrVectorProfile{}, fmt.Errorf("aux input length mismatch: %d != %d", len(aux), count)
	}
	totalStart := time.Now()
	inputSize := uint64(count * frElementBytes)
	auxBatch := normalizeBatch(aux, count)

	inputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-vector-input",
		Size:  inputSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuffer.Release()

	auxBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-vector-aux",
		Size:  inputSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create aux buffer: %w", err)
	}
	defer auxBuffer.Release()

	outputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-vector-output",
		Size:  inputSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create output buffer: %w", err)
	}
	defer outputBuffer.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-vector-staging",
		Size:  inputSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-vector-params",
		Size:  frVectorUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	uploadStart := time.Now()
	if err := k.device.Queue().WriteBuffer(inputBuffer, 0, flattenBatch(input)); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("write input buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(auxBuffer, 0, flattenBatch(auxBatch)); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("write aux buffer: %w", err)
	}

	params := make([]byte, frVectorUniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(op))
	binary.LittleEndian.PutUint32(params[8:], logCount)
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("write uniform buffer: %w", err)
	}

	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bls12-381-fr-vector-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuffer, Size: inputSize},
			{Binding: 1, Buffer: auxBuffer, Size: inputSize},
			{Binding: 2, Buffer: outputBuffer, Size: inputSize},
			{Binding: 3, Buffer: uniform, Size: frVectorUniformBytes},
		},
	})
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create bind group: %w", err)
	}
	defer bindGroup.Release()
	uploadElapsed := time.Since(uploadStart)

	kernelStart := time.Now()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("create command encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32((count+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(outputBuffer, 0, staging, 0, inputSize)

	cmd, err := encoder.Finish()
	if err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("submit: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("wait idle: %w", err)
	}
	kernelElapsed := time.Since(kernelStart)

	readbackStart := time.Now()
	data := make([]byte, inputSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, FrVectorProfile{}, fmt.Errorf("read output buffer: %w", err)
	}
	readbackElapsed := time.Since(readbackStart)

	return unflattenBatch(data, count), FrVectorProfile{
		Upload:   uploadElapsed,
		Kernel:   kernelElapsed,
		Readback: readbackElapsed,
		Total:    time.Since(totalStart),
	}, nil
}

func bitReverseLogCount(count int) (uint32, error) {
	if count <= 0 {
		return 0, fmt.Errorf("bit reverse requires non-zero input")
	}
	if count&(count-1) != 0 {
		return 0, fmt.Errorf("bit reverse requires power-of-two length, got %d", count)
	}
	return uint32(bits.Len(uint(count)) - 1), nil
}
