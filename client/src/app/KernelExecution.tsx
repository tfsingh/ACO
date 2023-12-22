const endpoint = process.env.NEXT_PUBLIC_ENDPOINT

export async function sendCuda(editorRef: any) {
    if (editorRef.current == null) {
        return;
    }

    let executionResult = editorRef.current.getValue();
}

export async function sendTriton(editorRef: any, email: string) {
    if (editorRef.current == null) {
        return;
    }

    const requestBody = {
        code: editorRef.current.getValue(),
        email: email,
    };

    try {
        const response = await fetch(endpoint + '/triton', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error('Failed to send Triton request');
        }

        return await response.text();
    } catch (error) {
        return "Error executing request";
    }
}
