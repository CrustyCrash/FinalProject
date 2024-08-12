# Use an official Alpine Linux as a base image
FROM debian:bookworm-slim

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    gfortran
    
# Set the working directory inside the container
WORKDIR /app

# Copy the entire project to the container
COPY . .

# Run CMake to configure the build system
RUN cmake .

# Build the project
RUN make
