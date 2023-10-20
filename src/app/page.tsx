"use client";

import Editor, { Monaco } from "@monaco-editor/react";
import React, { useRef, useState, useEffect } from 'react';
import { defaultCuda, defaultTriton, defaultLanguage, defaultResult } from "./constants";
import Login from "./Login";

export default function App(){
  const [selectedLanguage, setselectedLanguage] = useState<string>(() => {
    const storedLanguage = localStorage.getItem('selectedLanguage');
    return storedLanguage || defaultLanguage;
  });

  const [cudaCode, setCudaCode] = useState<string | undefined>(() => {
    const storedCudaCode = localStorage.getItem('cudaCode');
    return storedCudaCode || defaultCuda;
  });
  const [tritonCode, setTritonCode] = useState<string | undefined>(() => {
    const storedTritonCode = localStorage.getItem('tritonCode');
    return storedTritonCode || defaultTriton;
  });

  const [cudaResult, setCudaResult] = useState<string>(() => {
    const storedCudaResult = localStorage.getItem('cudaResult');
    return storedCudaResult || defaultResult;
  });
  const [tritonResult, setTritonResult] = useState<string>(() => {
    const storedTritonResult = localStorage.getItem('tritonResult');
    return storedTritonResult || defaultResult;
  });
  const editorRef = useRef<any>(null);

  function handleEditorDidMount(editor: any, monaco: any) {
    editorRef.current = editor;
  }

  useEffect(() => {
    if (selectedLanguage === "cuda") {
      localStorage.setItem('cudaCode', cudaCode || '');
      localStorage.setItem('cudaResult', cudaResult);
    } else if (selectedLanguage === "triton") {
      localStorage.setItem('tritonCode', tritonCode || '');
      localStorage.setItem('tritonResult', tritonResult);
    }
    localStorage.setItem('selectedLanguage', selectedLanguage);
  }, [cudaCode, cudaResult, tritonCode, tritonResult, selectedLanguage]);

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
    
      <div className="flex flex-col">
        <div className="flex flex-row">
        <div className="w-full bg-black-500 text-white text-lg py-1 px-4 ">
          Accelerated Computing, Online
        </div>
        <Login></Login>
        </div>
       
        <div>
          {selectedLanguage === "cuda" ? (
              <Editor
                height="100vh"
                width="70vw"
                language="cpp"
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
                language="python"
                value={tritonCode}
                theme="vs-dark"
                onMount={handleEditorDidMount}
                onChange={(newTritonCode: string | undefined) =>
                  setTritonCode(newTritonCode)
                }
              ></Editor>
            )}
        </div>
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
