"use client";
import Editor from "@monaco-editor/react";
import React, { useRef, useState, useEffect } from 'react';
import { defaultCuda, defaultTriton, defaultLanguage, defaultCudaResult, defaultTritonResult } from "./constants";
import { useSession, signIn, signOut } from "next-auth/react";
import Image from "next/image";
import githubLogo from ".//../../public/github-logo.png"

export default function App() {
  const { data: session } = useSession();

  const [selectedLanguage, setselectedLanguage] = useState<string>(() => {
    const storedLanguage = localStorage.getItem('selectedLanguage');
    return storedLanguage || defaultLanguage;
  });

  const [cudaCode, setCudaCode] = useState<string | undefined>(() => {
    const storedCudaCode = localStorage.getItem('cudaCode');
    return storedCudaCode === '__empty__' ? '' : storedCudaCode || defaultCuda;
  });

  const [tritonCode, setTritonCode] = useState<string | undefined>(() => {
    const storedTritonCode = localStorage.getItem('tritonCode');
    return storedTritonCode === '__empty__' ? '' : storedTritonCode || defaultTriton;
  });

  const [cudaResult, setCudaResult] = useState<string>(() => {
    const storedCudaResult = localStorage.getItem('cudaResult');
    return storedCudaResult === '__empty__' ? '' : storedCudaResult || defaultCudaResult;
  });

  const [tritonResult, setTritonResult] = useState<string>(() => {
    const storedTritonResult = localStorage.getItem('tritonResult');
    return storedTritonResult === '__empty__' ? '' : storedTritonResult || defaultTritonResult;
  });

  const editorRef = useRef<any>(null);

  function handleEditorDidMount(editor: any, monaco: any) {
    editorRef.current = editor;
  }

  useEffect(() => {
    if (selectedLanguage === "cuda") {
      localStorage.setItem('cudaCode', cudaCode === '' ? '__empty__' : cudaCode || '');
      localStorage.setItem('cudaResult', cudaResult === '' ? '__empty__' : cudaResult);
    } else if (selectedLanguage === "triton") {
      localStorage.setItem('tritonCode', tritonCode === '' ? '__empty__' : tritonCode || '');
      localStorage.setItem('tritonResult', tritonResult === '' ? '__empty__' : tritonResult);
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
    <div className="flex flex-col">
      <div className="flex flex-row">

        <div className="w-full bg-slate-900 text-white text-lg py-1 px-4">
          Accelerated Computing, Online
        </div>
        <div className="bg-slate-900">
          <Image src={githubLogo} height={50} width={50} alt="github" />
        </div>
        {!session?.user?.name ? (
          <button
            onClick={() => signIn()}
            type="button"
            className="border-2 border-emerald-500 btn btn-primary bg-slate-900 text-white py-1 px-4 w-5/12 text-base rounded"
          >
            Sign In
          </button>
        ) : (
          <div className="flex flex-row w-5/12">

            <select
              className="border-2 border-blue-500 text-center text-white py-1 px-4 w-5/12 text-base bg-slate-900 rounded-l"
              onChange={(e) => setselectedLanguage(e.target.value)}
              value={selectedLanguage}
            >
              <option value="cuda">CUDA</option>
              <option value="triton">Triton/Numba</option>
            </select>
            <button
              className="border-2 border-emerald-500 bg-slate-900 text-white w-1/3 py-1 px-4 text-base"
              onClick={sendFunction}
            >
              Run Kernel
            </button>
            <button
              onClick={() => signOut()}
              type="button"
              className="border-2 border-red-500 btn btn-primary text-white w-1/3 py-1 px-4 text-base bg-slate-900 rounded-r"
            >
              Sign Out
            </button>

          </div>
        )}

      </div>
      <div className="flex flex-row h-screen">
        <div>
          {selectedLanguage === "cuda" ? (
            <Editor
              height="100vh"
              width="71vw"
              language="cpp"
              value={cudaCode}
              theme="vs-dark"
              onMount={handleEditorDidMount}
              onChange={(newCudaCode: string | undefined) =>
                setCudaCode(newCudaCode)
              }
            />
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
            />
          )}
        </div>
        <pre
          className="text-xs text-zinc-300 pt-1 float-right font-mono overflow-y-auto"
          style={{ whiteSpace: "pre-wrap" }}
        >
          {result}
        </pre>
      </div>
    </div >
  )
};