import axios from "axios";
import type { AxiosInstance } from "axios";
import type {
  Document,
  QueryRequest,
  QueryResponse,
  ComparisonRequest,
  ComparisonResponse,
  UploadResponse,
  AvailableStrategies,
  RAGConfiguration,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class RAGLabAPI {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        "Content-Type": "application/json",
      },
      timeout: 120000, // 60 second timeout for long-running RAG operations
    });
  }

  // Document endpoints
  async uploadDocument(
    file: File,
    chunking_strategy: string,
    chunk_size: number = 1024,
    chunk_overlap: number = 200,
    session_id?: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("chunking_strategy", chunking_strategy);
    formData.append("chunk_size", chunk_size.toString());
    formData.append("chunk_overlap", chunk_overlap.toString());
    if (session_id) {
      formData.append("session_id", session_id);
    }

    const response = await this.client.post<UploadResponse>(
      "/api/v1/documents/upload",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 300000, // 5 minutes for document upload and initial processing
      }
    );

    return response.data;
  }

  async getDocuments(session_id?: string): Promise<Document[]> {
    const params = session_id ? { session_id } : {};
    const response = await this.client.get<Document[]>(
      "/api/v1/documents",
      { params }
    );
    return response.data;
  }

  async getDocument(doc_id: string): Promise<Document> {
    const response = await this.client.get<Document>(
      `/api/v1/documents/${doc_id}`
    );
    return response.data;
  }

  async deleteDocument(doc_id: string): Promise<void> {
    await this.client.delete(`/api/v1/documents/${doc_id}`);
  }

  // Strategy endpoints
  async getAvailableStrategies(): Promise<AvailableStrategies> {
    const response = await this.client.get<AvailableStrategies>(
      "/api/v1/documents/strategies/available"
    );
    return response.data;
  }

  // RAG Query endpoints
  async executeQuery(request: QueryRequest): Promise<QueryResponse> {
    const response = await this.client.post<QueryResponse>(
      "/api/v1/rag/query",
      request,
      {
        timeout: 120000, // 120 seconds for query execution (some techniques are slow)
      }
    );
    return response.data;
  }

  async comparePipelines(
    request: ComparisonRequest
  ): Promise<ComparisonResponse> {
    const response = await this.client.post<ComparisonResponse>(
      "/api/v1/rag/compare",
      request,
      {
        timeout: 120000, // 120 seconds for comparison (executes two queries in parallel)
      }
    );
    return response.data;
  }

  async validateTechniques(techniques: string[]): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const response = await this.client.post("/api/v1/rag/validate", techniques);
    return response.data;
  }

  // RAG Configuration
  async getRAGConfiguration(): Promise<RAGConfiguration> {
    const response = await this.client.get<RAGConfiguration>(
      "/api/v1/rag/configuration"
    );
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<{ healthy: boolean; message: string }> {
    const response = await this.client.get("/health");
    return response.data;
  }
}

export const api = new RAGLabAPI();
