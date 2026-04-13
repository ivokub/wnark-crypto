package metal

func BytesForGPULimbs(elements int, limbsPerElement int) int {
	return elements * limbsPerElement * 4
}

func BytesForField(elements int, shape FieldShape) int {
	return BytesForGPULimbs(elements, shape.GPULimbs)
}
