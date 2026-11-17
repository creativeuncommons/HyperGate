# Stage 1: Build Rust binary
FROM rust:1.80-alpine AS builder

WORKDIR /usr/src/hypergate
RUN apk add --no-cache musl-dev

COPY hypergate-rs/Cargo.toml hypergate-rs/Cargo.lock* ./
COPY hypergate-rs/src ./src

RUN cargo build --release --locked || cargo build --release

# Stage 2: Minimal Distroless / Scratch Runtime
FROM alpine:3.20 AS runtime

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app
COPY --from=builder /usr/src/hypergate/target/release/hypergate /app/hypergate
COPY config/model_pricing.json /app/config/model_pricing.json

ENV HYPERGATE_HOST="0.0.0.0"
ENV HYPERGATE_PORT="8080"
ENV HYPERGATE_CONFIG="/app/config/model_pricing.json"

EXPOSE 8080

ENTRYPOINT ["/app/hypergate"]
