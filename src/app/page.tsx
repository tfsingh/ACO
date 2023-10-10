"use client";

import Editor, { Monaco } from "@monaco-editor/react";
import React, { useRef, useState, useEffect } from 'react';

const defaultCuda = `// Here's some CUDA to get you started!
#include <iostream>
#include <cstdlib>

__global__ void addArrays(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Defines the size of the arrays
    int size = 5;

    // Allocates memory on host
    int *h_a, *h_b, *h_c;
    h_a = (int *)malloc(size * sizeof(int));
    h_b = (int *)malloc(size * sizeof(int));
    h_c = (int *)malloc(size * sizeof(int));

    // Initializes arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocates memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copies data from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Defines the grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launches the CUDA kernel
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);

    // Copies the result from device to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Prints the result
    for (int i = 0; i < size; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
`;

const App = () => {
  const [cudaCode, setCudaCode] = useState<string | undefined>(() => {
    const storedCudaCode = localStorage.getItem('cudaCode');
    return storedCudaCode || defaultCuda;
  });

  const [result, setResult] = useState<string>("[0, 3, 6, 9, 12]");
  const editorRef = useRef<any>(null);

  function handleEditorDidMount(editor: any, monaco: any) {
    editorRef.current = editor;
  }

  useEffect(() => {
    localStorage.setItem('cudaCode', cudaCode || '');
  }, [cudaCode]);

  function sendKernel() {
    if (editorRef.current == null) {
      return;
    }

    let executionResult = editorRef.current.getValue();
    
    setResult(executionResult);
  }

  return (
    <div className="flex flex-row">
      <div className="w-32px">
        <Editor
          height="100vh"
          width="70vw"
          defaultLanguage="cpp"
          value={cudaCode} 
          theme="vs-dark"
          onMount={handleEditorDidMount}
          onChange={(newCudaCode : string | undefined) => setCudaCode(newCudaCode)} 
        ></Editor>
      </div>
      
      <div className="flex flex-col w-screen h-screen">
        <button className="bg-blue-500 text-white text-lg py-1 px-4" onClick={sendKernel}>
          Execute Kernel
        </button> 
        <pre className="text-sm text-zinc-300 pt-1 float-right font-mono overflow-y-auto" style={{ whiteSpace: "pre-wrap" }}>&gt;&gt; {result}</pre>
      </div>
    </div>
  );
}

export default App;