# Hosting Architecture — ISPS System (CW Sections 3.6 & 3.7)

> **TODO**: Implement the full written explanation of the proposed hosting
> architecture and deployment strategy.

## Sections to cover

1. **Current architecture** — local Streamlit app, local ChromaDB, .env secrets
2. **Cloud deployment option** — Streamlit Community Cloud + environment variables
3. **Containerisation** — Dockerfile approach, `streamlit run` as entrypoint
4. **Data persistence** — ChromaDB volume mount or managed vector DB alternative
5. **Security** — API key handling, no secrets in git, `.gitignore` enforcement
6. **Scalability** — stateless app layer, persistent vector store
7. **CI/CD** — GitHub Actions for linting and testing
