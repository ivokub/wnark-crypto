export async function fetchShaderText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load shader ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}
