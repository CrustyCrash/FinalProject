# Use an official Alpine Linux as a base image
FROM alpine:latest

# Install necessary build tools
RUN apk update && apk add \
    build-base \
    cmake
    # Install additional dependencies if required
    # e.g., libgcc libstdc++

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project to the container
COPY . .

# Run CMake to configure the build system
RUN cmake .

# Build the project
RUN make

# Run the executable
CMD ["./boltzmann-openmp"]
