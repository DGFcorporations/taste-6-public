# taste-6

This repository contains the engine for TASTE variant 6, a self-editing AI agent designed to learn and enforce an individual developer's personal coding standards and preferences. It solves the problem of generic AI generation by extracting, curating, and injecting highly specific, personalized knowledge into AI tasks, ensuring the output aligns perfectly with the user's unique style and constraints.

**What's never here:** the owner's corpus (standards, decision records, rationale, persona-derived layers). Those live in the private repo and are the system's learned knowledge — publishing them would publish the person.

Each trial cycle: corpus pruned → keepers stay private → mechanics worth sharing land here via the publish decision.

## Stack
- **Language:** Python 3.11+
- **Core Frameworks:** FastAPI, Pydantic
- **AI & Embeddings:** fastembed
- **Integration:** Model Context Protocol (MCP)

## Key Features
- **Document Ingestion:** Extracts provisional standards and rules ("layers") from raw text.
- **Task Packets:** Compiles curated context and inferred standards into execution-ready packets.
- **Context Curation:** Maintains a bounded, dynamic "working set" of the most relevant rules.
- **Feedback Loop:** Logs owner decisions and automatically infers new standards from corrections.

## How to Run
The system provides a CLI for managing the agent's state and running the API server.

```bash
# Start the API server via granian or uvicorn
taste serve

# Extract rules from a document
taste ingest <file.md>

# View current learned standards
taste standards

# View the curated working set
taste working-set
```
