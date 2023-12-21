"use client";
import Editor from "@monaco-editor/react";
import React, { useRef, useEffect } from 'react';
import { defaultCuda, defaultTriton, defaultLanguage, defaultCudaResult, defaultTritonResult } from "./constants";
import { useSession } from "next-auth/react";

import useLocalStorageState from "./LocalStorageState";
import Header from "./Header";

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
      <Header
        session={session}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        sendFunction={sendFunction}
      />
      <div className="flex flex-row h-screen">
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