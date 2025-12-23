// RAG Technique Union Type
export const RAG_TECHNIQUES = {
  // Layer 1: Indexing Strategy (Mutually Exclusive)
  STANDARD_CHUNKING: "standard_chunking",
  PARENT_DOCUMENT: "parent_document",
  SEMANTIC_CHUNKING: "semantic_chunking",
  CONTEXTUAL_HEADERS: "contextual_headers",
  PROPOSITION_CHUNKING: "proposition_chunking",

  // Layer 2: Pipeline Components (Mix & Match)
  HYDE: "hyde",
  BASIC_RAG: "basic_rag",
  FUSION_RETRIEVAL: "fusion_retrieval",
  RERANKING: "reranking",
  CONTEXTUAL_COMPRESSION: "contextual_compression",

  // Layer 3: Advanced Controllers (Max One)
  SELF_RAG: "self_rag",
  CRAG: "crag",
  ADAPTIVE_RETRIEVAL: "adaptive_retrieval",
} as const;

export type RAGTechnique = typeof RAG_TECHNIQUES[keyof typeof RAG_TECHNIQUES];

// Document types
export interface Document {
  id: string;
  session_id: string;
  filename: string;
  status: "pending" | "processing" | "completed" | "failed";
  indexing_progress: number;
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  uploaded_at: string;
  error?: string;
}

// Chunk Info
export interface ChunkInfo {
  page: string | number;
  text: string;
  line_start?: number;
  line_end?: number;
  score?: number;
}

// Query Request/Response
export interface QueryRequest {
  document_id: string;
  query: string;
  techniques: RAGTechnique[];
  query_params?: Record<string, any>;
  session_id?: string;
}

export interface QueryResponse {
  response: string;
  retrieved_chunks: ChunkInfo[];
  result_id: string;
  scores: {
    latency_ms: number;
    token_count_est: number;
    [key: string]: any;
  };
}

// Pipeline Configuration
export interface PipelineConfig {
  techniques: RAGTechnique[];
  query_params?: Record<string, any>;
}

// Comparison Request/Response
export interface ComparisonRequest {
  document_id: string;
  query: string;
  pipeline_1: PipelineConfig;
  pipeline_2: PipelineConfig;
  session_id?: string;
}

export interface ComparisonMetrics {
  semantic_similarity: number;
  latency_diff_ms: number;
  interpretation: string;
}

export interface ComparisonResponse {
  pipeline_1_result: QueryResponse;
  pipeline_2_result: QueryResponse;
  comparison: ComparisonMetrics;
}

// API Response types
export interface UploadResponse {
  document_id: string;
  session_id: string;
  status: string;
}

export interface AvailableStrategy {
  name: string;
  description?: string;
  chunk_size_default?: number;
  chunk_overlap_default?: number;
}

export interface AvailableStrategies {
  strategies: Record<string, AvailableStrategy>;
  count: number;
}

// Application state
export type AppMode = "single" | "comparison";

export interface AppContextType {
  mode: AppMode;
  sessionId?: string;
  documentId?: string;
  document?: Document;
  setMode: (mode: AppMode) => void;
  setSessionId: (id: string) => void;
  setDocumentId: (id: string) => void;
  setDocument: (doc: Document) => void;
  reset: () => void;
}

// RAG Configuration
export interface TechniqueInfo {
  value: string;
  label: string;
  description: string;
  required: boolean;
  mutually_exclusive: boolean;
  mutually_exclusive_with: string[];
  requires: string[];
  enables: string[];
  warning?: string;
  default: boolean;
}

export interface LayerValidationRule {
  selection_type: string; // "single_required", "single_optional", "multi_optional"
  mutually_exclusive: boolean;
  conflicts: Array<{ techniques: string[]; reason: string }>;
  dependencies: Array<{ technique: string; requires: string[] }>;
}

export interface RAGDefaults {
  indexing_strategy: string;
  techniques: string[];
  chunk_size: number;
  chunk_overlap: number;
}

export interface RAGConfiguration {
  indexing_strategies: Record<string, Record<string, any>>;
  techniques: Record<string, TechniqueInfo[]>;
  validation_rules: Record<string, LayerValidationRule>;
  defaults: RAGDefaults;
}
