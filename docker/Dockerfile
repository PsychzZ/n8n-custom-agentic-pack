# =============================================================================
# n8n + Agentic Pack Custom Nodes
#
# n8n uses pnpm internally, so we can't use "npm install" inside n8n's dir.
# Instead, we copy our built dist/ directly into n8n's node_modules and
# symlink the shared peer dependencies (already present in n8n's own modules).
#
# Rebuild after code changes:
#   npm run build
#   docker compose -f docker/docker-compose.yml up -d --build
# =============================================================================

FROM n8nio/n8n:latest

USER root

# 1. Copy built package into n8n's node_modules
COPY package.json /usr/local/lib/node_modules/n8n-nodes-agentic-pack/package.json
COPY dist/        /usr/local/lib/node_modules/n8n-nodes-agentic-pack/dist/

# 2. Symlink peer dependencies from n8n's own node_modules so require() works
RUN N8N_MOD=/usr/local/lib/node_modules/n8n/node_modules \
    PKG=/usr/local/lib/node_modules/n8n-nodes-agentic-pack && \
    mkdir -p "$PKG/node_modules" && \
    for dep in n8n-workflow chromadb openai uuid; do \
      if [ -d "$N8N_MOD/$dep" ]; then \
        ln -sf "$N8N_MOD/$dep" "$PKG/node_modules/$dep"; \
        echo "Linked: $dep"; \
      else \
        echo "WARN: $dep not found in n8n modules, skipping"; \
      fi \
    done

# 3. Tell n8n where to find our package
ENV N8N_CUSTOM_EXTENSIONS=/usr/local/lib/node_modules/n8n-nodes-agentic-pack

USER node
