#!/bin/bash

# Docker Build Cache Optimization Script
# This script enables BuildKit and builds with optimal cache settings
# Usage: ./docker-build-cache.sh [dev|prod] [--no-cache]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="${1:-dev}"
NO_CACHE="${2}"
DOCKER_BUILDKIT=1

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Docker Build with Cache Optimization${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Validate build type
if [[ "$BUILD_TYPE" != "dev" && "$BUILD_TYPE" != "prod" ]]; then
    echo -e "${YELLOW}Invalid build type: $BUILD_TYPE${NC}"
    echo "Usage: ./docker-build-cache.sh [dev|prod] [--no-cache]"
    exit 1
fi

echo -e "\n${BLUE}Configuration:${NC}"
echo "BuildKit enabled:     ✓"
echo "Build type:           $BUILD_TYPE"

if [[ "$NO_CACHE" == "--no-cache" ]]; then
    echo "Cache usage:          ✗ (Force clean rebuild)"
else
    echo "Cache usage:          ✓ (Use cached layers)"
fi

echo ""

# Export BuildKit flag
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export BUILDKIT_PROGRESS=plain

# Build with appropriate compose file
if [[ "$BUILD_TYPE" == "dev" ]]; then
    COMPOSE_FILE="docker-compose.yml"
    TARGET_STAGE="development"
    echo -e "${GREEN}Building development image with Docker BuildKit...${NC}\n"
elif [[ "$BUILD_TYPE" == "prod" ]]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    TARGET_STAGE="production"
    echo -e "${GREEN}Building production image with Docker BuildKit...${NC}\n"
fi

# Build command
if [[ "$NO_CACHE" == "--no-cache" ]]; then
    docker-compose --file $COMPOSE_FILE build --no-cache --build-arg BUILDKIT_INLINE_CACHE=1 cuad-api
else
    docker-compose --file $COMPOSE_FILE build cuad-api
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo -e "${BLUE}To run the container:${NC}"
if [[ "$BUILD_TYPE" == "dev" ]]; then
    echo "  docker-compose up"
else
    echo "  docker-compose --file docker-compose.prod.yml up"
fi

echo ""
echo -e "${BLUE}To view build cache usage:${NC}"
echo "  docker buildx du"

echo ""
echo -e "${BLUE}To force a clean rebuild next time:${NC}"
echo "  ./docker-build-cache.sh $BUILD_TYPE --no-cache"
