FROM golang:latest

WORKDIR /app

COPY . .

RUN go build -o main .

FROM python:3.10

WORKDIR /app

RUN pip install modal

COPY --from=0 /app/main .

COPY base_kernel.py .

EXPOSE 8080

CMD modal token set --token-id $MODAL_TOKEN --token-secret $MODAL_SECRET; ./main
