"use client";
import React, { useRef, useEffect } from 'react';
import { defaultCuda, defaultTriton, defaultLanguage, defaultCudaResult, defaultTritonResult } from "./constants";
import { useSession } from "next-auth/react";
import useLocalStorageState from "./LocalStorageState";
import Header from "./Header";
import EditorResult from "./EditorResult"
import { sendTriton } from "./KernelExecution"

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

  const result = selectedLanguage === "cuda" ? cudaResult : tritonResult;
  const sendFunction = selectedLanguage === "cuda"
    ? () => { setCudaResult("CUDA not yet supported!") }
    : async () => {
      if (editorRef.current == null) {
        return;
      }
      setTritonResult("Executing kernel...");
      setTritonResult(await sendTriton(editorRef, session?.user?.email));
    };

  return (
    <div className="flex flex-col">
      <Header
        session={session}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        sendFunction={sendFunction}
      />
      <EditorResult
        handleEditorDidMount={handleEditorDidMount}
        selectedLanguage={selectedLanguage}
        setCudaCode={setCudaCode}
        setTritonCode={setTritonCode}
        cudaCode={cudaCode}
        tritonCode={tritonCode}
        result={result}
      />
    </div >
  )
};