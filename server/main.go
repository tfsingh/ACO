package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strings"
)

func tritonHandler(w http.ResponseWriter, r *http.Request) {
    code, err := ioutil.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error reading code from request", http.StatusBadRequest)
        return
    }

    tempFile, err := ioutil.TempFile("", "kernel-*.py")
    if err != nil {
        http.Error(w, "Error creating temporary file", http.StatusInternalServerError)
        return
    }
    defer os.Remove(tempFile.Name())

    baseKernel, err := ioutil.ReadFile("base_kernel.py")
    if err != nil {
        http.Error(w, "Error reading base_kernel.py", http.StatusInternalServerError)
        return
    }

    baseKernelStr := string(baseKernel)
    codeStr := string(code)
    codeStr = strings.Replace(codeStr, "\n", "\n    ", -1)
    baseKernelStr = strings.Replace(baseKernelStr, "# CODE", codeStr, 1)

    _, err = tempFile.Write([]byte(baseKernelStr))
    if err != nil {
        http.Error(w, "Error writing code to temporary file", http.StatusInternalServerError)
        return
    }

    /*
    fileContents, err := ioutil.ReadFile(tempFile.Name())
    if err != nil {
        http.Error(w, "Error reading temporary file", http.StatusInternalServerError)
        return
    }
    fmt.Printf("Temporary File Contents:\n%s\n", string(fileContents))*/

    cmd := exec.Command("modal", "run", "-q", tempFile.Name())
    output, err := cmd.CombinedOutput()
    if err != nil {
        fmt.Fprintf(w, "Error executing Triton command: %s\n", output)
        return
    }

	// Modal output trimming
	outputStr := string(output)

	lines := strings.Split(outputStr, "\n")

	if lines[len(lines) - 1] == "" {
		lines = lines[:len(lines) - 1]
	}
	n := len(lines)

	if n > 0 && lines[n-1] == "Stopping app - local entrypoint completed." {
		lines = lines[:n-1]
	} else if n > 2 && lines[n-1] == "Runner terminated." {
		lines = lines[:n-2]
		lines = append(lines, "Kernel execution took longer than limit (15s)")
	}

	output = []byte(strings.Join(lines, "\n"))

    fmt.Fprintf(w, "%s", output)
}


func cudaHandler(w http.ResponseWriter, r *http.Request) {
	return
}

func main() {
	http.HandleFunc("/triton", tritonHandler)
	http.HandleFunc("/cuda", cudaHandler)

	port := 8080
	fmt.Printf("Server listening on :%d\n", port)
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
	if err != nil {
		fmt.Printf("Error starting server: %s\n", err)
	}
}