# README

## Prerequisites

- [Nix](https://nixos.org/)
- [Ollama](https://ollama.com/) (with any of the following models downloaded: `llama3.2`, `llama3`, `mistral`, `deepseek-r1`)

## Install

```bash
ollama start # If service not running already
nix develop
```

### Using `direnv`

Copy the `.envrc.example` to `.envrc` and fill in the information then run `direnv allow .` in the repository root.

## Run

From inside the Nix development shell (started by `nix develop`).

```bash
python semantic_search.py -d ./pdfs/nke-10k-2023.pdf -q "How many distribution centers does Nike have in the US?"
```

# IDEA
- LLM as CLI tool, stdin/stdout, pipeable, previous input as system prompt/initial message, add rag inputs.
