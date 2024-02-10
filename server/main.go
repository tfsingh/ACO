package main

import (
    "context"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "bytes"
    "net/http"
    "os"
    "strconv"
    "time"
    "github.com/redis/go-redis/v9"
    // "github.com/joho/godotenv"
)

const MaxRequests = 30

type RequestBody struct {
	Code  string `json:"code"`
	Email string `json:"email"`
}

func rateLimit(email string) error {
	// envErr := godotenv.Load()
	// if envErr != nil {
	// 	return fmt.Errorf("Error loading .env file")
	// }

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
	w.Header().Set("Access-Control-Allow-Origin", "https://www.acceleratedcomputingonline.com")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusBadRequest)
		return
	}

	var requestBody RequestBody
	err = json.Unmarshal(body, &requestBody)
	if err != nil {
		http.Error(w, "Error decoding JSON", http.StatusBadRequest)
		return
	}

	email := requestBody.Email

	err = rateLimit(email)
	if err != nil {
		http.Error(w, "Reached rate limit on requests (resets every 24h).", http.StatusTooManyRequests)
		return
	}

	modalKey := os.Getenv("MODAL_KEY")
	endpoint := os.Getenv("MODAL_TRITON_ENDPOINT")

	payload := struct {
		Code string `json:"code"`
	}{
		Code: requestBody.Code,
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "Error marshaling JSON", http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(jsonStr))
	if err != nil {
		http.Error(w, "Error creating request to external service", http.StatusInternalServerError)
		return
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", modalKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "Error making request to modal", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading response from modal", http.StatusInternalServerError)
		return
	}

    var outputStr string
	str := string(body)
    if err := json.Unmarshal([]byte(str), &outputStr); err != nil {
		fmt.Println("Error unmarshalling:", err)
		return
	}
	fmt.Fprintf(w, str)
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
