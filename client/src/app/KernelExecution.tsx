export async function sendCuda(editorRef: any) {
    if (editorRef.current == null) {
        return;
    }

    let executionResult = editorRef.current.getValue();
}

export async function sendTriton(editorRef: any, email: string | null | undefined) {
    if (editorRef.current == null) {
        return;
    }

    const requestBody = {
        code: editorRef.current.getValue(),
        email: email,
    };

    try {
        const response = await fetch('http://localhost:8080/triton', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error('Failed to send Triton request');
        }

        return await response.json();
    } catch (error) {
        return "Error executing request";
    }
}
