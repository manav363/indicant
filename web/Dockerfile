# Build the terminal, serve it from nginx, proxy /api to the gateway.
#
# Two stages: node builds, nginx serves. The runtime image carries no node, no
# npm and no source — a static bundle behind a web server, which is the whole
# attack surface a SPA needs.

FROM node:20-alpine AS build
WORKDIR /app

# Copied first so a source-only change does not re-run the install layer.
COPY web/package.json web/package-lock.json* ./
RUN npm ci

COPY web/ ./
RUN npm run build

FROM nginx:1.27-alpine
# Replaces the default server block; ours sets CSP, caching and the /api proxy.
COPY infra/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
