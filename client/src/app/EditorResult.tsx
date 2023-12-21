import Editor from "@monaco-editor/react";

interface EditorResultProps {
    handleEditorDidMount: any;
    selectedLanguage: string;
    setCudaCode: React.Dispatch<React.SetStateAction<string | undefined>>;
    setTritonCode: React.Dispatch<React.SetStateAction<string | undefined>>;
    cudaCode: string;
    tritonCode: string;
    result: string;
}

const Header: React.FC<EditorResultProps> = ({
    handleEditorDidMount,
    selectedLanguage,
    setCudaCode,
    setTritonCode,
    cudaCode,
    tritonCode,
    result
}) => {

    return (
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
                className="text-xs text-zinc-300 pt-1 pl-1 float-right font-mono overflow-y-auto"
                style={{ whiteSpace: "pre-wrap" }}
            >
                {result}
            </pre>
        </div>
    );
};

export default Header;
