# base node image
FROM node:20 as base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
RUN corepack prepare pnpm@latest-8 --activate

# build the app
FROM base AS build-env
COPY . /app
WORKDIR /app

RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile
RUN pnpm run build

# install prod dependencies (if any)
FROM base AS prod-deps

COPY . /app
WORKDIR /app
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --prod --frozen-lockfile

# Finally, build the production image with minimal footprint
# Use tiny base image for running a node app
FROM gcr.io/distroless/nodejs20-debian12:nonroot
COPY --from=build-env /app/build /app/build
COPY --from=build-env /app/package.json /app/package.json
COPY --from=prod-deps /app/node_modules /app/node_modules
ENV NODE_ENV="production"
ENV ORIGIN=http://localhost:3000
WORKDIR /app
CMD ["build"]
