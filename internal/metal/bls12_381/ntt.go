package bls12381

import (
	"encoding/binary"
	"fmt"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/internal/metal"
)

const frNTTUniformBytes = 16

type FrNTTKernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
	vectorKernel   *FrVectorKernel
}

type FrNTTStageProfile struct {
	Upload   time.Duration
	Kernel   time.Duration
	Readback time.Duration
	Total    time.Duration
}

type FrNTTProfile struct {
	BitReverse FrVectorProfile
	Stages     FrNTTStageProfile
	Scale      FrVectorProfile
	Total      time.Duration
}

func ReadFrNTTShader() (string, error) {
	return metal.ReadShader("curves/bls12_381/fr_ntt.wgsl")
}

func NewFrNTTKernel(device *wgpu.Device) (*FrNTTKernel, error) {
	shaderText, err := ReadFrNTTShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bls12-381-fr-ntt-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bls12-381-fr-ntt-bgl",
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
		Label:            "bls12-381-fr-ntt-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bls12-381-fr-ntt-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "fr_ntt_stage_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	vectorKernel, err := NewFrVectorKernel(device)
	if err != nil {
		pipeline.Release()
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create vector kernel: %w", err)
	}
	return &FrNTTKernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
		vectorKernel:   vectorKernel,
	}, nil
}

func (k *FrNTTKernel) Close() {
	if k == nil {
		return
	}
	if k.vectorKernel != nil {
		k.vectorKernel.Close()
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

func (k *FrNTTKernel) RunStageProfiled(input, twiddles []metal.U32x8, m int) ([]metal.U32x8, FrNTTStageProfile, error) {
	count := len(input)
	if count == 0 {
		return nil, FrNTTStageProfile{}, nil
	}
	if count&(count-1) != 0 {
		return nil, FrNTTStageProfile{}, fmt.Errorf("ntt stage requires power-of-two count, got %d", count)
	}
	if m <= 0 || count%(2*m) != 0 {
		return nil, FrNTTStageProfile{}, fmt.Errorf("invalid stage width m=%d for count=%d", m, count)
	}
	if len(twiddles) != m {
		return nil, FrNTTStageProfile{}, fmt.Errorf("twiddle length mismatch: got=%d want=%d", len(twiddles), m)
	}
	totalStart := time.Now()
	dataSize := uint64(count * frElementBytes)
	twiddleSize := uint64(len(twiddles) * frElementBytes)

	inputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-ntt-input",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuffer.Release()

	twiddleBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-ntt-twiddles",
		Size:  twiddleSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create twiddle buffer: %w", err)
	}
	defer twiddleBuffer.Release()

	outputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-ntt-output",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create output buffer: %w", err)
	}
	defer outputBuffer.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-ntt-staging",
		Size:  dataSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bls12-381-fr-ntt-params",
		Size:  frNTTUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	uploadStart := time.Now()
	if err := k.device.Queue().WriteBuffer(inputBuffer, 0, flattenBatch(input)); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write input buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(twiddleBuffer, 0, flattenBatch(twiddles)); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write twiddle buffer: %w", err)
	}
	params := make([]byte, frNTTUniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(m))
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write uniform buffer: %w", err)
	}
	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bls12-381-fr-ntt-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuffer, Size: dataSize},
			{Binding: 1, Buffer: twiddleBuffer, Size: twiddleSize},
			{Binding: 2, Buffer: outputBuffer, Size: dataSize},
			{Binding: 3, Buffer: uniform, Size: frNTTUniformBytes},
		},
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create bind group: %w", err)
	}
	defer bindGroup.Release()
	uploadElapsed := time.Since(uploadStart)

	kernelStart := time.Now()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create command encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32(((count/2)+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(outputBuffer, 0, staging, 0, dataSize)
	cmd, err := encoder.Finish()
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("submit: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("wait idle: %w", err)
	}
	kernelElapsed := time.Since(kernelStart)

	readbackStart := time.Now()
	data := make([]byte, dataSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("read output buffer: %w", err)
	}
	readbackElapsed := time.Since(readbackStart)

	return unflattenBatch(data, count), FrNTTStageProfile{
		Upload:   uploadElapsed,
		Kernel:   kernelElapsed,
		Readback: readbackElapsed,
		Total:    time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) ForwardDITFullPathProfiled(input []metal.U32x8, stageTwiddles [][]metal.U32x8) ([]metal.U32x8, FrNTTProfile, error) {
	totalStart := time.Now()
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	state, bitReverse, err := k.vectorKernel.RunProfiled(FrVectorOpBitReverseCopy, input, nil, uint32(len(stageTwiddles)))
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("bit reverse input: %w", err)
	}
	var stages FrNTTStageProfile
	for _, twiddles := range stageTwiddles {
		var stageProfile FrNTTStageProfile
		state, stageProfile, err = k.RunStageProfiled(state, twiddles, len(twiddles))
		if err != nil {
			return nil, FrNTTProfile{}, err
		}
		stages.Upload += stageProfile.Upload
		stages.Kernel += stageProfile.Kernel
		stages.Readback += stageProfile.Readback
		stages.Total += stageProfile.Total
	}
	return state, FrNTTProfile{
		BitReverse: bitReverse,
		Stages:     stages,
		Total:      time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) InverseDITFullPathProfiled(input []metal.U32x8, inverseStageTwiddles [][]metal.U32x8, scale metal.U32x8) ([]metal.U32x8, FrNTTProfile, error) {
	totalStart := time.Now()
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	state, bitReverse, err := k.vectorKernel.RunProfiled(FrVectorOpBitReverseCopy, input, nil, uint32(len(inverseStageTwiddles)))
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("bit reverse inverse input: %w", err)
	}
	var stages FrNTTStageProfile
	for _, twiddles := range inverseStageTwiddles {
		var stageProfile FrNTTStageProfile
		state, stageProfile, err = k.RunStageProfiled(state, twiddles, len(twiddles))
		if err != nil {
			return nil, FrNTTProfile{}, err
		}
		stages.Upload += stageProfile.Upload
		stages.Kernel += stageProfile.Kernel
		stages.Readback += stageProfile.Readback
		stages.Total += stageProfile.Total
	}
	scaleBatch := make([]metal.U32x8, len(state))
	for i := range scaleBatch {
		scaleBatch[i] = scale
	}
	scaled, scaleProfile, err := k.vectorKernel.RunProfiled(FrVectorOpMulFactors, state, scaleBatch, 0)
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("scale inverse output: %w", err)
	}
	return scaled, FrNTTProfile{
		BitReverse: bitReverse,
		Stages:     stages,
		Scale:      scaleProfile,
		Total:      time.Since(totalStart),
	}, nil
}
