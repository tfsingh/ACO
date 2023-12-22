import Editor from "@monaco-editor/react";

interface EditorResultProps {
    handleEditorDidMount: any;
    selectedLanguage: string;
    setCudaCode: (value: string) => void;
    setTritonCode: (value: string) => void;
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
                        onChange={(newCudaCode: string | undefined) => {
                            if (newCudaCode) {
                                setCudaCode(newCudaCode)
                            }
                        }}
                        options={{ scrollbar: { vertical: "hidden" } }}
                    />
                ) : (
                    <Editor
                        height="100vh"
                        width="70vw"
                        language="python"
                        value={tritonCode}
                        theme="vs-dark"
                        onMount={handleEditorDidMount}
                        onChange={(newTritonCode: string | undefined) => {
                            if (newTritonCode) {
                                setTritonCode(newTritonCode)
                            }
                        }}
                        options={{ scrollbar: { vertical: "hidden" } }}
                    />
                )}
            </div>
            <pre
                className="text-xs text-zinc-300 pt-5 pl-2.5 float-right font-mono overflow-y-auto"
                style={{ whiteSpace: "pre-wrap" }}
            >
                {result}
            </pre>
        </div>
    );
};

export default Header;
