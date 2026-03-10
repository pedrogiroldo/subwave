# Repository Guidelines

## Project Structure & Module Organization

Project layout:

- `src/`: React + TypeScript UI code (`App.tsx`, `main.tsx`) and styles (`App.css`).
- `src/assets/`: UI assets used by the frontend.
- `public/`: Static files served by Vite.
- `src-tauri/`: Tauri (Rust) backend, configuration, and native assets.
- `src-tauri/src/`: Rust application logic and command handlers.
- `src-tauri/models/`: Speech model data (large binaries).
- `dist/` and `src-tauri/target/`: Build outputs (generated).

## Build, Test, and Development Commands

Key commands:

- `npm run dev`: Start the Vite dev server for the React UI.
- `npm run build`: Type-check (`tsc`) and build the UI bundle with Vite.
- `npm run preview`: Serve the production UI bundle locally.
- `npm run tauri -- dev`: Run the full Tauri desktop app in dev mode.
- `npm run tauri -- build`: Build the desktop app bundles.

## Coding Style & Naming Conventions

- TypeScript/TSX uses 2-space indentation, double quotes, and semicolons (see `src/main.tsx`).
- Prefer React function components and named exports.
- TypeScript is `strict` with `noUnusedLocals` and `noUnusedParameters` enabled; fix warnings before committing.
- No ESLint/Prettier config is present; keep formatting consistent with existing files.

## Testing Guidelines

- No test runner is configured yet.
- If you add tests, prefer `src/**/*.test.tsx` for UI with React Testing Library + Vitest.
- For Rust, add unit tests inside `src-tauri/src` with `#[cfg(test)]`.
- When you introduce tests, add scripts to `package.json` and update this guide.

## Commit & Pull Request Guidelines

- Commit history is short and informal; use concise, imperative summaries (e.g., `Add streaming transcription`).
- Avoid vague messages (e.g., `idk`).
- PRs should include a brief problem/solution description, steps to verify, and screenshots or clips for UI changes.

## Configuration & Assets

- Tauri settings live in `src-tauri/tauri.conf.json`.
- Speech models in `src-tauri/models/` are large; avoid adding or replacing them without discussion.

# Comandos cargo

- Os comandos cargo devem ser executados dentro do diretório src-tauri.
- Sempre que eu perguntar sobre algum erro em algum arquivo, use cargo check para anlisar a saída do cargo.
