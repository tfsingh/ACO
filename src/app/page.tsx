"use client";

import Editor, { Monaco } from "@monaco-editor/react";

const App = () => {

  return (
      <div>
        <Editor
          height="90vh"
          defaultLanguage="cpp"
          defaultValue="// some comment"
          ></Editor>
      </div>
  )
}
export default App;