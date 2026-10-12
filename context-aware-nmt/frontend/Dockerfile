# ============================================================
# Stage 1 — Build the production bundle
# ============================================================
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .
RUN npm run build

# ============================================================
# Stage 2 — Development image (hot-reload via Vite)
# ============================================================
FROM node:20-alpine AS dev

WORKDIR /app

ENV NODE_ENV=development \
    VITE_HOST=0.0.0.0 \
    VITE_PORT=5173

COPY package*.json ./
RUN npm install --no-audit --no-fund

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

# ============================================================
# Stage 3 — Production image (serve built bundle via Vite preview)
# ============================================================
FROM node:20-alpine AS prod

WORKDIR /app

ENV NODE_ENV=production \
    VITE_HOST=0.0.0.0 \
    VITE_PORT=4173

COPY --from=builder /app/dist ./dist
COPY package*.json ./
RUN npm ci --omit=dev --no-audit --no-fund || npm install --omit=dev --no-audit --no-fund

EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
