# RAG Lab Frontend

A modern, single-window TypeScript/React application for RAG lab experimentation.

## Features

- **Single Query Mode**: Test a single RAG configuration with a document
- **Comparison Mode**: Compare two different RAG pipelines side-by-side
- **Flexible Configuration**: Choose from multiple indexing strategies and techniques
- **Visual Results**: See retrieved chunks, page numbers, and latency metrics
- **Similarity Metrics**: Compare results with semantic similarity scoring

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend server running on `http://localhost:8000`

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── FileUpload.tsx
│   ├── TechniqueSelector.tsx
│   ├── QueryInput.tsx
│   ├── ResultsDisplay.tsx
│   ├── ChunkDisplay.tsx
│   ├── ComparisonView.tsx
│   ├── SingleView.tsx
│   └── ModeSwitcher.tsx
├── contexts/            # React Context
│   └── AppContext.tsx
├── hooks/               # Custom React hooks
│   ├── useDocumentUpload.ts
│   └── useRAGQuery.ts
├── services/            # API and types
│   ├── api.ts
│   └── types.ts
├── App.tsx
├── main.tsx
└── index.css
```

## Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## API Integration

The frontend communicates with the RAG Lab backend via REST API:

- **Document Upload**: `POST /api/v1/documents/upload`
- **Query**: `POST /api/v1/rag/query`
- **Comparison**: `POST /api/v1/rag/compare`
- **Strategies**: `GET /api/v1/documents/strategies/available`

## Technologies

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## Features Explained

### Single Query Mode

1. Upload a PDF document
2. Select indexing strategy (Layer 1)
3. Select retrieval techniques (Layer 2)
4. Optionally add orchestration (Layer 3)
5. Enter your query
6. View results with chunks and metrics

### Comparison Mode

1. Upload a PDF document (shared)
2. Configure Pipeline 1 (left side)
3. Configure Pipeline 2 (right side)
4. Enter query (same for both)
5. Execute comparison
6. See side-by-side results with similarity metrics

## Development Notes

- No persistence - refresh to reset everything
- API errors are displayed to the user
- Polling is used for document indexing status
- Context-based state management for simplicity

## License

See parent directory LICENSE
