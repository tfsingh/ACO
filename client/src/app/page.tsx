"use client";
import Editor from "@monaco-editor/react";
import React, { useRef, useState, useEffect } from 'react';
import { defaultCuda, defaultTriton, defaultLanguage, defaultCudaResult, defaultTritonResult } from "./constants";
import { useSession, signIn, signOut } from "next-auth/react";
import Image from "next/image";
import githubLogo from ".//../../public/github-logo.png"
import useLocalStorageState from "./LocalStorageState";

export default function App() {
  const { data: session } = useSession();

  const [selectedLanguage, setSelectedLanguage] = useLocalStorageState('selectedLanguage', defaultLanguage);
  const [cudaCode, setCudaCode] = useLocalStorageState('cudaCode', defaultCuda);
  const [tritonCode, setTritonCode] = useLocalStorageState('tritonCode', defaultTriton);
  const [cudaResult, setCudaResult] = useLocalStorageState('cudaResult', defaultCudaResult);
  const [tritonResult, setTritonResult] = useLocalStorageState('tritonResult', defaultTritonResult);

  const editorRef = useRef<any>(null);

  function handleEditorDidMount(editor: any, monaco: any) {
    editorRef.current = editor;
  }

  useEffect(() => {
    const languageVariables: { [key: string]: { code: string | undefined; result: string } } = {
      cuda: { code: cudaCode, result: cudaResult },
      triton: { code: tritonCode, result: tritonResult },
    };

    const selectedLanguageVariables = languageVariables[selectedLanguage] || {};
    const { code = '', result = '' } = selectedLanguageVariables;

    localStorage.setItem('selectedLanguage', selectedLanguage);
    localStorage.setItem(`${selectedLanguage}Code`, code === '' ? '__empty__' : code);
    localStorage.setItem(`${selectedLanguage}Result`, result === '' ? '__empty__' : result);
  }, [cudaCode, cudaResult, tritonCode, tritonResult, selectedLanguage]);

  function sendCuda() {
    if (editorRef.current == null) {
      return;
    }

    let executionResult = editorRef.current.getValue();
    setCudaResult(executionResult);
  }

  async function sendTriton() {
    if (editorRef.current == null) {
      return;
    }

    try {
      setTritonResult("Executing kernel...")
      const response = await fetch('http://localhost:8080/triton', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/plain',
        },
        body: editorRef.current.getValue(),
      });

      if (!response.ok) {
        throw new Error('Failed to send Triton request');
      }

      const data = await response.json();

      setTritonResult(data.result);
    } catch (error) {
      setTritonResult("Error executing request")
    }
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
          <a href="https://github.com/tfsingh/aconline"><Image src={githubLogo} height={50} width={50} alt="github" /></a>
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
              onChange={(e) => setSelectedLanguage(e.target.value)}
              value={selectedLanguage}
            >
              <option value="triton">Triton/Numba</option>
              <option value="cuda">CUDA</option>
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
              width="71vw"
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