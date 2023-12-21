export async function sendCuda(editorRef: any) {
    if (editorRef.current == null) {
        return;
    }

    let executionResult = editorRef.current.getValue();
}

export async function sendTriton(editorRef: any) {
    if (editorRef.current == null) {
        return;
    }

    try {
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

    } catch (error) {
        return "Error executing request"
    }
}