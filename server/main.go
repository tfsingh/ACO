package main

import (
    "context"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
    "os/exec"
    "strconv"
    "strings"
    "time"
    "github.com/redis/go-redis/v9"
    //"github.com/joho/godotenv"
)

const MaxRequests = 30

type RequestBody struct {
	Code  string `json:"code"`
	Email string `json:"email"`
}

func rateLimit(email string) error {
	/*envErr := godotenv.Load()
	if envErr != nil {
		return fmt.Errorf("Error loading .env file")
	}*/

	var ctx = context.Background()
	redisEndpoint := os.Getenv("REDIS_ENDPOINT")
	redisPassword := os.Getenv("REDIS_PASSWORD")
    whitelistKey := os.Getenv("WHITELIST_KEY")

	rdb := redis.NewClient(&redis.Options{
		Addr:     redisEndpoint,
		Password: redisPassword,
		DB:       0,
	})

    isMember, err := rdb.SIsMember(context.Background(), whitelistKey, email).Result()
    if err != nil {
        return fmt.Errorf("Error reading whitelist: %v", err)
    } else if !isMember {
        return fmt.Errorf("User is not whitelisted")
    }

	key := "rate_limit:" + email
	expiration := 24 * time.Hour

	val, err := rdb.Get(ctx, key).Result()

	if err == redis.Nil {
		err := rdb.Set(ctx, key, 1, expiration).Err()
		if err != nil {
			return fmt.Errorf("Error setting rate limit: %v", err)
		}
	} else if err != nil {
		return fmt.Errorf("Error getting rate limit: %v", err)
	} else {
		count, err := strconv.Atoi(val)
		if err != nil {
			return fmt.Errorf("Error converting count to integer: %v", err)
		}

		if count >= MaxRequests {
			return fmt.Errorf("Request limit exceeded for email: %s", email)
		}

		err = rdb.Incr(ctx, key).Err()
		if err != nil {
			return fmt.Errorf("Error incrementing rate limit: %v", err)
		}
	}

	return nil
}

func tritonHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Access-Control-Allow-Origin", "acceleratedcomputingonline.com")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

    body, err := ioutil.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error reading code from request", http.StatusBadRequest)
        return
    }

    var requestBody RequestBody
	err = json.Unmarshal(body, &requestBody)
	if err != nil {
		http.Error(w, "Error decoding JSON", http.StatusBadRequest)
		return
	}

    code := requestBody.Code
	email := requestBody.Email

    err = rateLimit(email)
	if err != nil {
		fmt.Println("Error in rate limiting:", err)
        fmt.Fprintf(w, "Reached rate limit on requests (resets every 24h).")
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
    codeStr = strings.Replace(codeStr, "\n", "\n        ", -1)
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
    fmt.Printf("Temporary File Contents:\n%s\n", string(fileContents))
    */

    cmd := exec.Command("modal", "run", "-q", tempFile.Name())
    output, err := cmd.CombinedOutput()
    if err != nil {
        fmt.Fprintf(w, "Error executing Triton command: %s\n", output)
        return
    }

    outputStr := string(output)
    if strings.HasSuffix(outputStr, "Stopping app - local entrypoint completed.\n") {
        outputStr = strings.TrimSuffix(outputStr, "Stopping app - local entrypoint completed.\n")
    }

    fmt.Fprintf(w, outputStr)
}

func cudaHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "CUDA not supported")
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
