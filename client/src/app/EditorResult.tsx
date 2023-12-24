import React from 'react';
import Editor from '@monaco-editor/react';

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
                <Editor
                    height="100vh"
                    width="69.07vw"
                    language={selectedLanguage === 'cuda' ? 'cpp' : 'python'}
                    value={selectedLanguage === 'cuda' ? cudaCode : tritonCode}
                    theme="vs-dark"
                    onMount={handleEditorDidMount}
                    onChange={(newCode: string | undefined) => {
                        if (newCode) {
                            selectedLanguage === 'cuda' ? setCudaCode(newCode) : setTritonCode(newCode);
                        }
                    }}
                    options={{ scrollbar: { verticalScrollbarSize: 0 } }}
                />
            </div>
            <div className="border-l-2 border-gray-600 text-xs text-zinc-300 pt-5 pl-5 pr-5 float-right font-mono overflow-y-auto whitespace-pre-wrap">
                {result}
            </div>
        </div>
    );
};

export default Header;
