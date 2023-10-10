"use client";

import Editor, { Monaco } from "@monaco-editor/react";
import React, { useRef, useState, useEffect } from 'react';

const defaultCuda = `// Here's some CUDA to get you started!
// ... (your CUDA code)
`;

const defaultTriton = `# Here's some Python code for Triton
# ... (your Python code)
`;

const App = () => {
  const [selectedLanguage, setselectedLanguage] = useState<string>(() => {
    const storedLanguage = localStorage.getItem('selectedLanguage');
    return storedLanguage || "cuda";
  });

  const [cudaCode, setCudaCode] = useState<string | undefined>(() => {
    const storedCudaCode = localStorage.getItem('cudaCode');
    return storedCudaCode || defaultCuda;
  });
  const [tritonCode, setTritonCode] = useState<string | undefined>(() => {
    const storedTritonCode = localStorage.getItem('tritonCode');
    return storedTritonCode || defaultTriton;
  });

  const [cudaResult, setCudaResult] = useState<string>(">> [0, 3, 6, 9, 12]");
  const [tritonResult, setTritonResult] = useState<string>(">> [0, 3, 6, 9, 12]");

  const editorRef = useRef<any>(null);

  function handleEditorDidMount(editor: any, monaco: any) {
    editorRef.current = editor;
  }

  useEffect(() => {
    if (selectedLanguage === "cuda") {
      localStorage.setItem('cudaCode', cudaCode || '');
    } else if (selectedLanguage === "triton") {
      localStorage.setItem('tritonCode', tritonCode || '');
    }
    localStorage.setItem('selectedLanguage', selectedLanguage);
  }, [cudaCode, tritonCode, selectedLanguage]);

  function sendCuda() {
    if (editorRef.current == null) {
      return;
    }

    let executionResult = editorRef.current.getValue();
    setCudaResult(executionResult);
  }

  function sendTriton() {
    if (editorRef.current == null) {
      return;
    }

    let executionResult = editorRef.current.getValue();
    setTritonResult(executionResult);
  }

  const result = selectedLanguage === "cuda" ? cudaResult : tritonResult;
  const sendFunction = selectedLanguage === "cuda" ? sendCuda : sendTriton;

  return (
    <div className="flex flex-row">
      <div>
      {selectedLanguage === "cuda" ? (
          <Editor
            height="100vh"
            width="70vw"
            defaultLanguage="cpp"
            value={cudaCode}
            theme="vs-dark"
            onMount={handleEditorDidMount}
            onChange={(newCudaCode: string | undefined) =>
              setCudaCode(newCudaCode)
            }
          ></Editor>
        ) : (
          <Editor
            height="100vh"
            width="70vw"
            defaultLanguage="python"
            value={tritonCode}
            theme="vs-dark"
            onMount={handleEditorDidMount}
            onChange={(newTritonCode: string | undefined) =>
              setTritonCode(newTritonCode)
            }
          ></Editor>
        )}
      </div>
      <div className="flex flex-col w-screen h-screen">
        <div className="flex flex-row">
          <select
            className="bg-green-500 text-white text-lg py-1 px-4 w-1/2 rounded"
            onChange={(e) => setselectedLanguage(e.target.value)}
            value={selectedLanguage}>
            <option value="cuda">CUDA</option>
            <option value="triton">Triton</option>
          </select>
          <button
            className="bg-blue-500 text-left text-white text-lg py-1 px-4 w-1/2 rounded"
            onClick={sendFunction}
          >
            Execute Kernel
          </button>
        </div>
        <pre
          className="text-sm text-zinc-300 pt-1 float-right font-mono overflow-y-auto"
          style={{ whiteSpace: "pre-wrap" }}
        >
          {result}
        </pre>
      </div>   
    </div>
  );
}

export default App;
